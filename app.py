import os, glob, time
import numpy as np, pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, roc_curve)
from tensorflow.keras.applications.resnet50 import preprocess_input

DIR = "/content/drive/MyDrive/BrainTumorProject/"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
TEST_DIRS = ["/content/dataset/Testing", "/content/drive/MyDrive/archive (1)/Testing"]
MODEL_FILES = {os.path.basename(p): p for p in sorted(glob.glob(DIR + "*.keras"))}

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="wide")

@st.cache_resource
def load(path):
    return tf.keras.models.load_model(
        path, custom_objects={"preprocess_input": preprocess_input}, safe_mode=False)

def predict_one(model, img):
    arr = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
    model(arr, training=False)                       # warm-up call (not timed)
    t = time.perf_counter()
    probs = model(arr, training=False).numpy()[0]
    return probs, (time.perf_counter() - t) * 1000

def find_test_dir():
    for d in TEST_DIRS:
        if os.path.isdir(d):
            return d
    return None

def run_benchmark(path):
    model = load(path)
    ds = tf.keras.utils.image_dataset_from_directory(
        find_test_dir(), image_size=(224, 224), batch_size=32,
        label_mode="int", shuffle=False)
    names = ds.class_names
    n = len(names)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=0)
    y_pred = probs.argmax(1)

    cm = confusion_matrix(y_true, y_pred, labels=range(n))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(n), zero_division=0)
    spec = []
    for i in range(n):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp
        fn = cm[i].sum() - tp; tn = cm.sum() - tp - fp - fn
        spec.append(tn / (tn + fp))
    per_class = pd.DataFrame({"Class": names, "Precision": p, "Recall (Sensitivity)": r,
                              "Specificity": spec, "F1-score": f, "Test images": s})

    # ROC curves (one class vs the rest)
    rocs = {}
    for i, c in enumerate(names):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), probs[:, i])
        rocs[c] = (fpr, tpr)

    # Latency: single image
    x1 = np.random.randint(0, 255, (1, 224, 224, 3)).astype("float32")
    for _ in range(20): model(x1, training=False)
    ts = []
    for _ in range(100):
        t0 = time.perf_counter(); model(x1, training=False).numpy()
        ts.append((time.perf_counter() - t0) * 1000)
    # Throughput: batch of 32
    xb = np.random.randint(0, 255, (32, 224, 224, 3)).astype("float32")
    model(xb, training=False)
    t0 = time.perf_counter()
    for _ in range(10): model(xb, training=False).numpy()
    thr = 320 / (time.perf_counter() - t0)

    summary = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": p.mean(), "Macro Recall": r.mean(), "Macro F1": f.mean(),
        "ROC-AUC (OvR)": roc_auc_score(y_true, probs, multi_class="ovr"),
        "Test images": len(y_true), "Misclassified": int((y_true != y_pred).sum()),
        "Latency mean (ms)": float(np.mean(ts)), "Latency median (ms)": float(np.median(ts)),
        "Latency p95 (ms)": float(np.percentile(ts, 95)),
        "Throughput (img/s)": thr,
        "Parameters (M)": model.count_params() / 1e6,
        "File size (MB)": os.path.getsize(path) / 1e6,
    }
    return {"summary": summary, "per_class": per_class, "cm": cm, "names": names, "rocs": rocs}

def show_results(res):
    summ = pd.DataFrame({n: r["summary"] for n, r in res.items()}).T
    st.subheader("Summary of all models")
    st.dataframe(summ.round(4), use_container_width=True)
    st.download_button("Download summary as CSV", summ.to_csv().encode(), "benchmark_summary.csv")
    st.caption("Accuracy / F1 / ROC-AUC comparison")
    st.bar_chart(summ[["Accuracy", "Macro F1", "ROC-AUC (OvR)"]])

    for name, r in res.items():
        s = r["summary"]
        st.divider()
        st.header(name)
        a = st.columns(4)
        a[0].metric("Accuracy", f"{s['Accuracy']*100:.2f}%")
        a[1].metric("Macro F1", f"{s['Macro F1']:.4f}")
        a[2].metric("ROC-AUC", f"{s['ROC-AUC (OvR)']:.4f}")
        a[3].metric("Misclassified", f"{s['Misclassified']} / {s['Test images']}")
        b = st.columns(4)
        b[0].metric("Latency mean", f"{s['Latency mean (ms)']:.1f} ms")
        b[1].metric("Latency p95", f"{s['Latency p95 (ms)']:.1f} ms")
        b[2].metric("Throughput", f"{s['Throughput (img/s)']:.1f} img/s")
        b[3].metric("Model size", f"{s['File size (MB)']:.1f} MB")

        st.subheader("Per-class metrics")
        st.dataframe(r["per_class"].round(4), use_container_width=True)

        left, right = st.columns(2)
        names, cm = r["names"], r["cm"]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
        fig.tight_layout(); left.pyplot(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        for c, (fpr, tpr) in r["rocs"].items():
            ax.plot(fpr, tpr, label=c)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves (one class vs rest)"); ax.legend()
        fig.tight_layout(); right.pyplot(fig); plt.close(fig)

# ------------------------------ PAGE ------------------------------
st.title("Brain Tumor MRI Classifier")
st.caption("For educational purposes only. Not a medical diagnosis tool.")

if not MODEL_FILES:
    st.error("No .keras model files found in " + DIR)
    st.stop()

tab1, = st.tabs(["Predict"])

with tab1:
    choice = st.selectbox("Model", list(MODEL_FILES))
    file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
        c1, c2 = st.columns(2)
        c1.image(img, caption="Uploaded MRI", use_container_width=True)
        if c2.button("Predict", type="primary"):
            with st.spinner("Running the model..."):
                probs, ms = predict_one(load(MODEL_FILES[choice]), img)
            idx = int(probs.argmax())
            c2.metric("Predicted class", CLASSES[idx])
            c2.metric("Confidence", f"{probs[idx]*100:.2f}%")
            c2.metric("Inference time", f"{ms:.1f} ms")
            dfp = pd.DataFrame({"Class": CLASSES, "Probability (%)": probs * 100})
            c2.dataframe(dfp.round(2), use_container_width=True)
            c2.bar_chart(dfp.set_index("Class"))

