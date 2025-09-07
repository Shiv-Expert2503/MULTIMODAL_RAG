# Sensor Fault Detection: An End-to-End MLOps Project

This project demonstrates a **complete, production-grade MLOps pipeline** for a real-world industrial challenge: detecting faults in **sensor data from semiconductor wafers**. It proves how raw data can be transformed into a **deployed, scalable, and investor-ready AI solution**.  

📸 Snapshot: `sensor_fault_app_interface_snapshot.png`  
(A preview of the web app interface showing CSV upload, data preview, and prediction results.)

---

## 🌟 Key Features
- **Interactive Web App:** Upload CSVs, preview & edit data, and generate predictions in real time.  
- **Dynamic Data Handling:** In-browser preview ensures users can check and adjust inputs before running predictions.  
- **Instant Prediction:** Powered by an optimized **XGBoost model** wrapped in Flask APIs.  
- **Downloadable Results:** Export predictions as a new CSV for external use.  
- **One-Click Deployment:** Hosted live on **Render** with auto-deploy enabled on each `git push`.  

---

## ⚙️ Tech Stack
- **Backend:** Python, Flask, Gunicorn  
- **Frontend:** HTML, CSS, JavaScript  
- **ML & Data:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn  
- **Deployment:** Render, Git  

---

## 🔄 MLOps Pipeline
The project follows **professional-grade MLOps principles**:  

1. **Data Ingestion & Transformation** → Modular pipelines for clean preprocessing.  
2. **Model Training** → `training_pipeline.py` ensures reproducibility.  
3. **Prediction Pipeline** → `prediction_pipeline.py` guarantees consistent serving.  
4. **Artifact Management** → Models and preprocessors saved as `.pkl` files for easy versioning.  
5. **Deployment as API** → RESTful Flask API wrapped into a lightweight web app.  
6. **Continuous Deployment** → Auto-triggered Render builds keep deployment up-to-date with Git.  

---

## ⚖️ Model Performance & Limitations
- The dataset is **highly imbalanced** (≈94% “healthy” sensors, 6% “faulty”).  
- The deployed XGBoost model demonstrates **robust end-to-end integration**, but naturally skews towards the majority class.  

### ✅ Responsible AI Note
While this iteration focuses on **deployment architecture**, we are fully aware of techniques to handle imbalance:  
- **Data Balancing:** SMOTE, ADASYN, or undersampling methods.  
- **XGBoost Tuning:** `scale_pos_weight`, learning rate & tree-depth optimization.  
- **Evaluation Metrics:** F1-Score, Precision-Recall AUC, instead of just accuracy.  

These methods are **planned for future iterations** to further enhance predictive accuracy.  

---

## 🔮 Future Roadmap
- Integrate **SMOTE balancing** for fairer training data.  
- Deploy **ensemble models** to improve recall on rare faulty cases.  
- Add **Docker + CI/CD** pipeline for enterprise scaling.  
- Explore **cloud-native deployment** (AWS/GCP) for production.  

---

## 💡 FAQs
**Q: Can the model handle real industrial data streams?**  
Yes — the modular pipeline supports both batch CSV uploads and can be adapted for streaming data.  

**Q: What happens if the dataset is imbalanced?**  
We use **XGBoost with imbalance-aware tuning**, and future updates include **SMOTE-based balancing**.  

**Q: Is this just a demo or production-ready?**  
The system is **production-deployable** today, with Render-hosted CI/CD. Scaling options (Docker/K8s) are part of the roadmap.  

---

## 🔗 Related Concepts & Synonyms
- Sensor fault detection → *wafer sensor anomaly detection*, *industrial IoT fault prediction*, *manufacturing defect detection*.  
- MLOps → *end-to-end ML pipeline*, *model deployment workflow*, *production AI pipeline*.  
- XGBoost → *gradient boosting trees*, *imbalanced classification optimizer*.  

---

## 📌 Notes for Investors & End-Users
- **Deployment-first demo** — this is not just a notebook but a **live app**.  
- The **MLOps pipeline** is modular, reproducible, and auto-deployable.  
- Proven ability to take a raw dataset → **fully working cloud-deployed AI solution**.  
- Future improvements ensure performance grows alongside scalability.  

---

## 🚀 Resources
- **Live App Demo:** [🌐 View Deployment](https://sensor-fault-0bkp.onrender.com/)  
- **Code Repository:** [💻 GitHub Repository](https://github.com/Shiv-Expert2503/Sensor_Fault)  
