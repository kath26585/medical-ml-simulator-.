import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris, load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# --- PAGE CONFIG ---
st.set_page_config(page_title=" Medical ML Model Generator", layout="wide")

# ==========================================
# PAGE 1: HOME / ML MODEL GENERATOR
# ==========================================
def generator_page():
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio("", ["Generate Synthetic Data", "Upload Dataset"], index=0)

    st.sidebar.subheader("Feature Configuration")
    features_input = st.sidebar.text_input("Features (comma-separated)", "Age, BMI, Systolic BP, Diastolic BP, Cholesterol, Fasting Blood Sugar")

    st.sidebar.subheader("Classes")
    classes_input = st.sidebar.text_input("Classes (comma-separated)", "Low Risk, Moderate Risk, High Risk")

    st.sidebar.subheader("Class-Specific Settings")
    for cls in classes_input.split(","):
        cls_name = cls.strip()
        # Gamit ang popover para maglabas ng settings kapag clinick
        with st.sidebar.popover(f"⚙️ {cls_name} Settings"):
            st.write(f"Configure settings for **{cls_name}** class:")
            st.number_input(f"{cls_name} Threshold", value=0.5, key=f"thresh_{cls_name}")
            st.checkbox("Enable Alerts", value=True, key=f"alert_{cls_name}")
            st.slider("Priority Level", 0, 10, 5, key=f"priority_{cls_name}")
            if st.button("Save Changes", key=f"save_{cls_name}"):
                st.success(f"✅ Settings saved for {cls_name}!")

    st.sidebar.subheader("Number of Samples")
    n_samples = st.sidebar.slider("", 100, 20000, 10000)

    st.sidebar.subheader("Test Size (%)")
    test_size = st.sidebar.slider("", 10, 50, 30)

    st.sidebar.subheader("Cross-Validation Settings")
    cv_enabled = st.sidebar.checkbox("Enable K-Fold Cross-Validation", value=True)
    n_folds = st.sidebar.slider("Number of Folds (K)", 2, 10, 5)
    st.sidebar.caption("Higher K means more thorough validation but slower training")

    st.sidebar.subheader("Random State Control")
    random_state = st.sidebar.number_input("Random Seed", value=42)

    st.sidebar.subheader("Data Preprocessing")
    scaler_method = st.sidebar.selectbox("Feature Scaling Method", ["Standard Scaler", "MinMax Scaler"])
    stratified_split = st.sidebar.checkbox("Stratified Train-Test Split", value=True)

    st.sidebar.subheader("Class Imbalance Handling")
    imbalance_handling = st.sidebar.selectbox("", ["None", "SMOTE", "Class Weights"])

    st.sidebar.subheader("Performance vs Speed Trade-off")
    training_mode = st.sidebar.slider("Training Mode", 0, 10, 5, format="%d", help="Fast - Balanced - Thorough")

    st.sidebar.subheader("Model Hyperparameters")
    max_iter = st.sidebar.number_input("Max Iterations (Iterative Models)", value=1000)
    n_estimators = st.sidebar.number_input("Number of Estimators (Ensemble Models)", value=100)

    run_button = st.sidebar.button("🚀 Generate Data & Train Models", use_container_width=True, type="primary")

    # --- TITLE PLACEHOLDER (Mawawala ito habang nagta-train) ---
    title_placeholder = st.empty()
    title_placeholder.title("📈  Medical ML Model Generator")

    if run_button:
        # --- ITAGO MUNA ANG TITLE AT SLIDERS HABANG NAG-TRAIN ---
        title_placeholder.empty()

        features_list = [f.strip() for f in features_input.split(",")]
        classes_list = [c.strip() for c in classes_input.split(",")]
        n_classes = len(classes_list)
        n_features = len(features_list)

        # --- 1. GENERATE DATA ---
        with st.spinner("Generating synthetic medical data..."):
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=4,
                n_redundant=1,
                n_classes=n_classes,
                random_state=random_state,
                flip_y=0.05,
                weights=[0.5, 0.3, 0.2] if n_classes == 3 else None
            )
            df = pd.DataFrame(X, columns=features_list)
            df['Risk Level'] = [classes_list[i] for i in y]

        X_data = df.drop('Risk Level', axis=1)
        y_data = df['Risk Level']
        
        if stratified_split:
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size/100, random_state=random_state, stratify=y_data)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size/100, random_state=random_state)

        scaler = StandardScaler() if scaler_method == "Standard Scaler" else MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if imbalance_handling == "SMOTE":
            smote = SMOTE(random_state=random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=max_iter, random_state=random_state, class_weight='balanced' if imbalance_handling == 'Class Weights' else None),
            "RidgeClassifier": RidgeClassifier(random_state=random_state),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced' if imbalance_handling == 'Class Weights' else None),
            "AdaBoostClassifier": AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state),
            "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced' if imbalance_handling == 'Class Weights' else None),
            "SVC": SVC(random_state=random_state, probability=True, class_weight='balanced' if imbalance_handling == 'Class Weights' else None),
            "LinearSVC": LinearSVC(random_state=random_state, max_iter=max_iter, class_weight='balanced' if imbalance_handling == 'Class Weights' else None),
            "GaussianNB": GaussianNB(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "MLPClassifier": MLPClassifier(max_iter=max_iter, random_state=random_state)
        }

        # --- TRAINING LOOP WITH DONUT (GINITNA AT PINALIIT) ---
        results = []
        loading_placeholder = st.empty()
        results_placeholder = st.empty()

        def create_donut_progress(progress, current_idx, total_models):
            fig = go.Figure(data=[go.Pie(
                values=[max(0, progress * 100), max(0, (1 - progress) * 100)],
                hole=0.75,
                sort=False,
                direction='clockwise',
                marker=dict(colors=["#57E24A", '#e6e9ed']),
                showlegend=False,
                textinfo='none'
            )])
            fig.update_layout(
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=200, width=200, # PINALIIT NA ANG SIZE
                annotations=[
                    dict(
                        text=f"<b style='font-size:24px; color: #4A90E2;'>{int(progress * 100)}%</b><br><span style='font-size:12px; color: #555;'>Training model {current_idx}/{total_models}...</span>",
                        x=0.5, y=0.5, showarrow=False, xanchor='center', yanchor='middle'
                    )
                ]
            )
            return fig

        start_time_global = time.time()
        for i, (name, model) in enumerate(models.items()):
            current_progress = (i) / len(models)
            
            with loading_placeholder.container():
                # GINITNA SA PAMAMAGITAN NG COLUMNS
                c1, c2, c3 = st.columns([2, 6, 2])
                with c2:
                    donut_fig = create_donut_progress(current_progress, i+1, len(models))
                    st.plotly_chart(donut_fig, use_container_width=False)
                    st.markdown("<div style='text-align: center; font-size: 16px; margin-top: 5px;'>Training models... Please wait.</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; color: #555; font-size: 14px;'>Model {i+1} of {len(models)}</div>", unsafe_allow_html=True)
                
                elapsed = time.time() - start_time_global
                if i > 0:
                    avg_time = elapsed / (i + 1)
                    remaining = len(models) - (i + 1)
                    eta = int(avg_time * remaining)
                    eta_text = f"⏱️ Estimated time remaining: {eta}s"
                else:
                    eta_text = "⏱️ Estimating time..."
                st.markdown(f"<div style='text-align: center; color: #777; font-size: 13px;'>{eta_text}</div>", unsafe_allow_html=True)

            start_time = time.time()
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
                train_time = time.time() - start_time
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                cv_scores = []
                if cv_enabled:
                    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='accuracy')
                
                results.append({
                    "Model": name, "Accuracy": round(acc, 4), "Precision": round(prec, 4),
                    "Recall": round(rec, 4), "F1 Score": round(f1, 4), "Training Time (s)": round(train_time, 4),
                    "Status": "Success", "Object": model, "Predictions": y_pred, "Probabilities": y_prob,
                    "Scaler": scaler, "CV_Scores": cv_scores
                })
            except Exception:
                results.append({
                    "Model": name, "Accuracy": 0, "Precision": 0, "Recall": 0, "F1 Score": 0,
                    "Training Time (s)": 0, "Status": "Failed", "Object": None, "Predictions": [],
                    "Probabilities": [], "Scaler": None, "CV_Scores": []
                })

        loading_placeholder.empty()
        res_df = pd.DataFrame(results)

        # --- IBALIK NA ANG TITLE ---
        title_placeholder.title("📈 Medical ML Model Generator")

        # ==========================================
        # SHOW RESULTS WITH UNIFORM BOX STYLING
        # ==========================================

                # --- BOX 1: DATASET SPLIT INFORMATION (BAGONG GRADIENT COLORS) ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #4f46e5; font-weight: 600; font-size: 16px;">📊 Dataset Split Information</span>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 16px; padding: 20px; text-align: center; color: white; box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);">
                <h2 style="margin: 0; font-size: 32px;">{len(df)}</h2>
                <p style="margin: 0; font-size: 14px; opacity: 0.9;">Total Samples</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f472b6 0%, #ec4899 100%); border-radius: 16px; padding: 20px; text-align: center; color: white; box-shadow: 0 4px 10px rgba(236, 72, 153, 0.3);">
                <h2 style="margin: 0; font-size: 32px;">{len(X_train)} <span style="font-size: 18px; opacity: 0.9;">({100-test_size}%)</span></h2>
                <p style="margin: 0; font-size: 14px; opacity: 0.9;">Training Samples</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%); border-radius: 16px; padding: 20px; text-align: center; color: white; box-shadow: 0 4px 10px rgba(20, 184, 166, 0.3);">
                <h2 style="margin: 0; font-size: 32px;">{len(X_test)} <span style="font-size: 18px; opacity: 0.9;">({test_size}%)</span></h2>
                <p style="margin: 0; font-size: 14px; opacity: 0.9;">Testing Samples</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
                      # --- BOX 2: MODEL COMPARISON (BAGONG INDIGO HEADER) ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #4f46e5; font-weight: 600; font-size: 16px;">📋 Model Comparison</span>
            </div>
        """, unsafe_allow_html=True)
        
        html_lines = []
        html_lines.append('<div style="overflow-x: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">')
        html_lines.append('<table style="width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px; background-color: #ffffff; color: #333333;">')
        html_lines.append('<thead><tr style="background-color: #4f46e5; color: white; text-align: left;">')
        html_lines.append('<th style="padding: 12px 15px;">MODEL</th>')
        html_lines.append('<th style="padding: 12px 15px;">ACCURACY</th>')
        html_lines.append('<th style="padding: 12px 15px;">PRECISION</th>')
        html_lines.append('<th style="padding: 12px 15px;">RECALL</th>')
        html_lines.append('<th style="padding: 12px 15px;">F1-SCORE</th>')
        html_lines.append('<th style="padding: 12px 15px;">TRAINING TIME (S)</th>')
        html_lines.append('<th style="padding: 12px 15px;">STATUS</th>')
        html_lines.append('<th style="padding: 12px 15px;">ACTIONS</th>')
        html_lines.append('</tr></thead><tbody>')

        for idx, row in res_df.iterrows():
            bg_color = '#f8f9fa' if idx % 2 == 0 else '#ffffff'
            status_color = '#10b981' if row['Status'] == 'Success' else '#ef4444' # Soft Green & Soft Red
            html_lines.append(f'<tr style="border-bottom: 1px solid #e9ecef; background-color: {bg_color};">')
            html_lines.append(f'<td style="padding: 12px 15px; font-weight: bold;">{row["Model"]}</td>')
            html_lines.append(f'<td style="padding: 12px 15px;">{row["Accuracy"]:.4f}</td>')
            html_lines.append(f'<td style="padding: 12px 15px;">{row["Precision"]:.4f}</td>')
            html_lines.append(f'<td style="padding: 12px 15px;">{row["Recall"]:.4f}</td>')
            html_lines.append(f'<td style="padding: 12px 15px;">{row["F1 Score"]:.4f}</td>')
            html_lines.append(f'<td style="padding: 12px 15px;">{row["Training Time (s)"]:.4f}</td>')
            html_lines.append(f'<td style="padding: 12px 15px;"><span style="background-color: {status_color}; color: white; padding: 2px 12px; border-radius: 12px; font-weight: bold; font-size: 12px; display: inline-block;">{row["Status"]}</span></td>')
            html_lines.append(f'<td style="padding: 12px 15px;"><div style="background-color: #4f46e5; color: white; width: 30px; height: 30px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">⬇</div></td>')
            html_lines.append('</tr>')

        html_lines.append('</tbody></table></div>')
        st.markdown("\n".join(html_lines), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        best_model_row = res_df.loc[res_df['Accuracy'].idxmax()]

               # --- BOX 3: BEST MODEL PERFORMANCE (BAGONG KULAY NG CARDS) ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #4f46e5; font-weight: 600; font-size: 16px;">🏆 Best Model Performance</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; gap: 20px; flex-wrap: wrap; padding: 5px 0;">
            <div style="background: #f8f9fa; padding: 10px 20px; border-radius: 8px; border: 1px solid #e5e7eb; flex: 1; min-width: 120px;">
                <div style="font-size: 12px; color: #6b7280; font-weight: 600;">MODEL</div>
                <div style="font-size: 22px; font-weight: bold; color: #4f46e5;">{best_model_row['Model']}</div>
            </div>
            <div style="background: #f8f9fa; padding: 10px 20px; border-radius: 8px; border: 1px solid #e5e7eb; flex: 1; min-width: 120px;">
                <div style="font-size: 12px; color: #6b7280; font-weight: 600;">ACCURACY</div>
                <div style="font-size: 22px; font-weight: bold;">{best_model_row['Accuracy']:.4f}</div>
            </div>
            <div style="background: #f8f9fa; padding: 10px 20px; border-radius: 8px; border: 1px solid #e5e7eb; flex: 1; min-width: 120px;">
                <div style="font-size: 12px; color: #6b7280; font-weight: 600;">TRAINING TIME</div>
                <div style="font-size: 22px; font-weight: bold;">{best_model_row['Training Time (s)']}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 4: PERFORMANCE METRICS SUMMARY ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">📊 Performance Metrics Summary</span>
            </div>
        """, unsafe_allow_html=True)
        display_df = res_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Training Time (s)", "Status"]]
        fig_metrics = px.bar(display_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score']), 
                             x='Model', y='value', color='variable', barmode='group', title="Model Performance Metrics Comparison")
        st.plotly_chart(fig_metrics, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

                       # --- BOX 5: CONFUSION MATRICES (VIOLET THEME & SQUARE) ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div class="header-ribbon">🌀 Confusion Matrices</div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2)
        for idx, row in res_df.iterrows():
            if row['Status'] == 'Success':
                with cols[idx % 2]:
                    cm = confusion_matrix(y_test, row['Predictions'])
                    
                    # --- BAGONG KULAY AT HUGIS (PARISUKAT) ---
                    fig_cm = px.imshow(
                        cm, 
                        labels=dict(x="Predicted", y="Actual"), 
                        x=classes_list, 
                        y=classes_list, 
                        text_auto=True, 
                        color_continuous_scale="Viridis", # <--- Pinalitan ng Violet/Purple (Viridis)
                        aspect='equal' # <--- Pinalitan para maging parisukat (Square) katulad ng image
                    )
                    fig_cm.update_layout(
                        title=f"<b>{row['Model']}</b><br>Accuracy: {row['Accuracy']:.4f}", # <--- Dinagdag ang Accuracy
                        height=500, # Medyo binaba ang height para magkasya sa square aspect
                        margin=dict(l=20, r=20, t=60, b=20) 
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        # --- BOX 6: LEARNING CURVES ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">📈 Learning Curves (All Models)</span>
            </div>
        """, unsafe_allow_html=True)
        all_success_models = res_df[res_df['Status'] == 'Success']['Model'].tolist()
        lc_cols = st.columns(2)
        for idx, model_name in enumerate(all_success_models):
            with lc_cols[idx % 2]:
                try:
                    model_obj = res_df[res_df['Model'] == model_name]['Object'].values[0]
                    train_sizes, train_scores, test_scores = learning_curve(
                        model_obj, X_train_scaled, y_train, cv=n_folds, n_jobs=-1, 
                        train_sizes=np.linspace(0.1, 1.0, 5), random_state=random_state
                    )
                    train_mean = np.mean(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)
                    fig_lc, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(train_sizes, train_mean, label='Train Score', color='blue')
                    ax.plot(train_sizes, test_mean, label='Cross-Validation Score', color='orange')
                    ax.fill_between(train_sizes, train_mean - np.std(train_scores, axis=1), train_mean + np.std(train_scores, axis=1), alpha=0.1, color='blue')
                    ax.fill_between(train_sizes, test_mean - np.std(test_scores, axis=1), test_mean + np.std(test_scores, axis=1), alpha=0.1, color='orange')
                    ax.set_title(f"Learning Curve: {model_name}")
                    ax.set_xlabel("Training Examples")
                    ax.set_ylabel("Score")
                    ax.legend(loc="best")
                    st.pyplot(fig_lc)
                    plt.close(fig_lc)
                except Exception:
                    pass
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 7: FEATURE VISUALIZATION ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">📌 Feature Visualization</span>
            </div>
        """, unsafe_allow_html=True)
        col_3d1, col_3d2, col_3d3 = st.columns(3)
        with col_3d1:
            x_axis = st.selectbox("X-Axis Feature", features_list, index=0)
        with col_3d2:
            y_axis = st.selectbox("Y-Axis Feature", features_list, index=1)
        with col_3d3:
            z_axis = st.selectbox("Z-Axis Feature", features_list, index=2)
        fig_3d = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color='Risk Level', opacity=0.7, title="Feature Visualization (3D)")
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 8: DATASET STATISTICS ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">📊 Dataset Statistics</span>
            </div>
        """, unsafe_allow_html=True)
        col_orig, col_scaled = st.columns(2)
        with col_orig:
            st.write("**Original Dataset Statistics**")
            st.dataframe(df.drop('Risk Level', axis=1).describe().loc[['mean', 'std', 'min', 'max']].round(4), use_container_width=True)
        with col_scaled:
            st.write("**Scaled Dataset Statistics**")
            scaled_df_full = pd.DataFrame(np.vstack([X_train_scaled, X_test_scaled]), columns=features_list)
            st.dataframe(scaled_df_full.describe().loc[['mean', 'std', 'min', 'max']].round(4), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 9: FEATURE CORRELATION HEATMAP ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">🔥 Feature Correlation Heatmap</span>
            </div>
        """, unsafe_allow_html=True)
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.drop('Risk Level', axis=1).corr(), annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
        st.pyplot(fig_corr)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 10: CLASS DISTRIBUTION ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">📊 Class Distribution</span>
            </div>
        """, unsafe_allow_html=True)
        train_counts = pd.Series(y_train).value_counts().reindex(classes_list).fillna(0)
        test_counts = pd.Series(y_test).value_counts().reindex(classes_list).fillna(0)
        class_dist_df = pd.DataFrame({'Training': train_counts, 'Test': test_counts})
        fig_dist = px.bar(class_dist_df, barmode='group', title="Class Distribution (Training vs Test Set)")
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 11: FEATURE IMPORTANCE ---
        tree_models = ['RandomForestClassifier', 'ExtraTreesClassifier']
        best_tree = None
        for m in tree_models:
            if m in res_df['Model'].values and res_df[res_df['Model'] == m]['Status'].values[0] == 'Success':
                best_tree = m
                break
        if best_tree:
            st.markdown("""
            <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                    <span style="color: #2563eb; font-weight: 600; font-size: 16px;">⭐ Feature Importance</span>
                </div>
            """, unsafe_allow_html=True)
            model_obj = res_df[res_df['Model'] == best_tree]['Object'].values[0]
            importances = model_obj.feature_importances_
            imp_df = pd.DataFrame({'Feature': features_list, 'Importance': importances})
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title=f"Feature Importance - {best_tree}")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 12: ROC CURVES ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">📈 ROC Curves & AUC Scores</span>
            </div>
        """, unsafe_allow_html=True)
        fig_roc = go.Figure()
        for idx, row in res_df.iterrows():
            if row['Status'] == 'Success' and row['Probabilities'] is not None:
                for i, cls in enumerate(classes_list):
                    fpr, tpr, _ = roc_curve((y_test == cls).astype(int), row['Probabilities'][:, i])
                    roc_auc = auc(fpr, tpr)
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{row['Model']} (AUC={roc_auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500)
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 13: PRECISION-RECALL CURVES ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">📉 Precision-Recall Curves</span>
            </div>
        """, unsafe_allow_html=True)
        fig_pr = go.Figure()
        for idx, row in res_df.iterrows():
            if row['Status'] == 'Success' and row['Probabilities'] is not None:
                precision, recall, _ = precision_recall_curve((y_test == classes_list[-1]).astype(int), row['Probabilities'][:, -1])
                ap = average_precision_score((y_test == classes_list[-1]).astype(int), row['Probabilities'][:, -1])
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f"{row['Model']} (AP={ap:.3f})"))
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=500)
        st.plotly_chart(fig_pr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 14: RADAR CHART ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">🕸️ Model Comparison Radar Chart</span>
            </div>
        """, unsafe_allow_html=True)
        top_5 = res_df.nlargest(5, 'Accuracy')
        radar_data = top_5.copy()
        max_time = radar_data['Training Time (s)'].max()
        radar_data['Speed'] = 1 - (radar_data['Training Time (s)'] / (max_time + 1e-6))
        fig_radar = go.Figure()
        for idx, row in radar_data.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1 Score'], row['Speed']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Speed'],
                fill='toself', name=row['Model']
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 15: CROSS-VALIDATION RESULTS ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">📊 Cross-Validation Results</span>
            </div>
        """, unsafe_allow_html=True)
        if cv_enabled:
            cv_data = []
            for idx, row in res_df.iterrows():
                if row['Status'] == 'Success' and len(row['CV_Scores']) > 0:
                    cv_data.append({
                        'Model': row['Model'],
                        'CV Mean ± STD': f"{np.mean(row['CV_Scores']):.3f} ± {np.std(row['CV_Scores']):.3f}",
                        **{f'Fold {i+1}': f"{score:.3f}" for i, score in enumerate(row['CV_Scores'])}
                    })
            if cv_data:
                st.dataframe(pd.DataFrame(cv_data), use_container_width=True)
        else:
            st.info("K-Fold Cross-Validation is currently disabled in the sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 16: ENSEMBLE MODEL ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">🤝 Ensemble Model (Top 3 Models)</span>
            </div>
        """, unsafe_allow_html=True)
        top_3 = res_df.nlargest(3, 'Accuracy')['Model'].tolist()
        if len(top_3) == 3:
            estimators = [(name, models[name]) for name in top_3]
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensemble.fit(X_train_scaled, y_train)
            y_pred_ens = ensemble.predict(X_test_scaled)
            y_prob_ens = ensemble.predict_proba(X_test_scaled)
            acc_ens = accuracy_score(y_test, y_pred_ens)
            st.info(f"**Models Combined:** {', '.join(top_3)}\n**Voting Method:** Soft Voting\n**Ensemble Accuracy:** {acc_ens:.4f}")
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=["Ensemble"], y=[acc_ens], name="Ensemble", marker_color='green'))
            for mod in top_3:
                val = res_df[res_df['Model'] == mod]['Accuracy'].values[0]
                fig_comp.add_trace(go.Bar(x=[mod], y=[val], name=mod))
            fig_comp.update_layout(title="Ensemble vs Individual Models", yaxis_title="Accuracy")
            st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 17: PREDICTION CONFIDENCE DISTRIBUTION ---
        if len(top_3) == 3 and 'y_prob_ens' in locals():
            st.markdown("""
            <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                    <span style="color: #2563eb; font-weight: 600; font-size: 16px;">📊 Prediction Confidence Distribution</span>
                </div>
            """, unsafe_allow_html=True)
            confidences = np.max(y_prob_ens, axis=1)
            fig_conf = px.histogram(x=confidences, nbins=20, title="Ensemble Model Prediction Confidence Distribution")
            st.plotly_chart(fig_conf, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 18: DOWNLOADS ---
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="border-bottom: 1px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #2563eb; font-weight: 600; font-size: 16px;">⬇️ Downloads</span>
            </div>
        """, unsafe_allow_html=True)
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Original Dataset", data=csv, file_name="medical_dataset.csv", mime="text/csv", type="primary")
        with dcol2:
            scaled_df_full = pd.DataFrame(np.vstack([X_train_scaled, X_test_scaled]), columns=features_list)
            scaled_csv = scaled_df_full.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Scaled Dataset", data=scaled_csv, file_name="medical_dataset_scaled.csv", mime="text/csv", type="primary")

        st.subheader("Download Trained Models")
        for idx, row in res_df.iterrows():
            if row['Status'] == 'Success':
                col_a, col_b = st.columns(2)
                with col_a:
                    model_bytes = pickle.dumps(row['Object'])
                    st.download_button(f"🧠 {row['Model']} Model", data=model_bytes, file_name=f"{row['Model']}.pkl", mime="application/octet-stream")
                with col_b:
                    scaler_bytes = pickle.dumps(row['Scaler'])
                    st.download_button(f"⚖️ {row['Model']} Scaler", data=scaler_bytes, file_name=f"{row['Model']}_scaler.pkl", mime="application/octet-stream")

        st.success("✅ All models trained successfully! You can now download the datasets and models.")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("👈 Configure your parameters on the left sidebar and click **Generate Data & Train Models** to start!")

# ==========================================
# PAGE 2: ALGORITHM EDUCATION (HUMAN/MEDICAL THEME)
# ==========================================
def education_page():
    st.title("🎓 Machine Learning Algorithm Education")
    st.info("Welcome to the Algorithm Education page! This interactive guide helps you understand various machine learning algorithms used in medical and health classification tasks.")
    
    st.write("### How to Use This Guide")
    st.write("1. Select an algorithm from the dropdown menu below.")
    st.write("2. Explore its characteristics and implementation details.")
    st.write("3. Test the algorithm on built-in datasets via the interactive demo.")

    algos = ["Gaussian Naive Bayes", "Linear Support Vector Classification", "Support Vector Classification", 
             "Multi-layer Perceptron", "Random Forest Classifier", "Extra Trees Classifier", 
             "K-Nearest Neighbors", "Ridge Classifier", "Multinomial Naive Bayes", "AdaBoost Classifier"]
    
    selected_algo = st.selectbox("🔍 Select Algorithm", algos)

    # --- ALGORITHM 1 ---
    if selected_algo == "Gaussian Naive Bayes":
        st.subheader("Gaussian Naive Bayes")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Description")
            st.write("A probabilistic classifier based on Bayes' theorem with strong independence assumptions between medical features. Assumes clinical features follow a Gaussian (normal) distribution.")
            st.markdown("#### Advantages")
            st.write("✅ Simple and fast for real-time patient triage")
            st.write("✅ Works well with small medical datasets")
            st.write("✅ Good for high-dimensional clinical data")
            st.write("✅ Performs well when medical indicators are normally distributed")
            st.markdown("#### Common Use Cases")
            st.write("🔴 Patient health risk classification")
            st.write("🔴 Medical symptom diagnosis")
            st.write("🔴 Real-time patient monitoring systems")
        with col2:
            st.markdown("#### Disadvantages")
            st.write("⚠️ Assumes independence of medical symptoms (often unrealistic)")
            st.write("⚠️ Limited by Gaussian distribution assumption in health data")
            st.write("⚠️ May underperform when vital signs are highly correlated")
        st.subheader("Mathematical Formula"); st.latex(r"P(y|x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i|y)}{P(x_1, \dots, x_n)}")
        st.subheader("Implementation Example"); st.code("from sklearn.naive_bayes import GaussianNB\n\ngnb = GaussianNB()\ngnb.fit(X_train, y_train)\ny_pred = gnb.predict(X_test)", language="python")
        st.subheader("Key Parameters"); st.write("**var_smoothing** : Portion of the largest variance added for stability")
        
        st.subheader("🎮 Interactive Demo")
        dset = st.selectbox("Select Dataset", ["Iris", "Digits"], key="gnb_ds")
        if st.button("▶️ Run Demo", key="gnb_btn"):
            data = load_iris() if dset == "Iris" else load_digits()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = MinMaxScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            clf = GaussianNB().fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                fig_cm = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"), x=data.target_names, y=data.target_names, text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.write("**Learning Curve**")
                train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
                fig_lc, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
                ax.set_title("Learning Curve"); ax.set_xlabel("Training examples"); ax.set_ylabel("Score"); ax.legend(loc="best")
                st.pyplot(fig_lc); plt.close(fig_lc)

    # --- ALGORITHM 2 to 10 ---
    elif selected_algo == "Linear Support Vector Classification":
        st.subheader("Linear Support Vector Classification (LinearSVC)")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Description"); st.write("Fits a linear decision boundary (hyperplane) to separate patient risk categories. Optimized for linear clinical data separation.")
            st.markdown("#### Advantages"); st.write("✅ Extremely efficient for large patient databases"); st.write("✅ Works well when clinical features are linearly separable"); st.write("✅ Less prone to overfitting on medical data")
        with col2:
            st.markdown("#### Disadvantages"); st.write("⚠️ Assumes linear relationships between medical features"); st.write("⚠️ Cannot handle complex non-linear patient patterns"); st.write("⚠️ Requires careful scaling of medical variables (BMI, BP, etc.)")
        st.subheader("Mathematical Formula"); st.latex(r"\min_{w, b} \frac{1}{2} ||w||^2 + C \sum \max(0, 1 - y_i (w \cdot x_i - b))")
        st.subheader("Implementation Example"); st.code("from sklearn.svm import LinearSVC\n\nclf = LinearSVC(C=1.0, max_iter=10000)\nclf.fit(X_train, y_train)", language="python")
        st.subheader("Key Parameters"); st.write("**C** : Regularization parameter")
        
        st.subheader("🎮 Interactive Demo")
        dset = st.selectbox("Select Dataset", ["Iris", "Digits"], key="lin_ds")
        if st.button("▶️ Run Demo", key="lin_btn"):
            data = load_iris() if dset == "Iris" else load_digits()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = MinMaxScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            clf = LinearSVC(C=1.0, max_iter=10000).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                fig_cm = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"), x=data.target_names, y=data.target_names, text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.write("**Learning Curve**")
                train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
                fig_lc, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
                ax.set_title("Learning Curve"); ax.set_xlabel("Training examples"); ax.set_ylabel("Score"); ax.legend(loc="best")
                st.pyplot(fig_lc); plt.close(fig_lc)

    elif selected_algo == "Support Vector Classification":
        st.subheader("Support Vector Classification (SVC with RBF Kernel)")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Description"); st.write("Uses the Radial Basis Function (RBF) kernel to map complex patient data into higher dimensions and find non-linear decision boundaries.")
            st.markdown("#### Advantages"); st.write("✅ Excellent at handling complex, non-linear medical patterns"); st.write("✅ Robust to outliers in patient data"); st.write("✅ Powerful when health conditions are not linearly separable")
        with col2:
            st.markdown("#### Disadvantages"); st.write("⚠️ Computationally expensive for large electronic health records"); st.write("⚠️ Highly sensitive to kernel and regularization parameters")
        st.subheader("Mathematical Formula (RBF Kernel)"); st.latex(r"K(x, x') = \exp(-\gamma ||x - x'||^2)")
        st.subheader("Implementation Example"); st.code("from sklearn.svm import SVC\n\nclf = SVC(kernel='rbf', C=1.0, gamma='scale')\nclf.fit(X_train, y_train)", language="python")
        st.subheader("Key Parameters"); st.write("**C** : Regularization parameter"); st.write("**gamma** : Kernel coefficient")
        
        st.subheader("🎮 Interactive Demo")
        dset = st.selectbox("Select Dataset", ["Iris", "Digits"], key="svc_ds")
        if st.button("▶️ Run Demo", key="svc_btn"):
            data = load_iris() if dset == "Iris" else load_digits()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = MinMaxScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            clf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                fig_cm = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"), x=data.target_names, y=data.target_names, text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.write("**Learning Curve**")
                train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
                fig_lc, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
                ax.set_title("Learning Curve"); ax.set_xlabel("Training examples"); ax.set_ylabel("Score"); ax.legend(loc="best")
                st.pyplot(fig_lc); plt.close(fig_lc)

    elif selected_algo == "Multi-layer Perceptron":
        st.subheader("Multi-layer Perceptron (MLPClassifier)")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Description"); st.write("A feedforward artificial neural network consisting of input, hidden, and output layers. Learns complex relationships between patient vital signs using backpropagation.")
            st.markdown("#### Advantages"); st.write("✅ Can learn highly complex non-linear functions (e.g., gene-protein interactions)"); st.write("✅ Capable of handling classification and regression for disease prediction")
        with col2:
            st.markdown("#### Disadvantages"); st.write("⚠️ Difficult and time-consuming to tune for medical data"); st.write("⚠️ Prone to overfitting without proper regularization")
        st.subheader("Implementation Example"); st.code("from sklearn.neural_network import MLPClassifier\n\nclf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)\nclf.fit(X_train, y_train)", language="python")
        st.subheader("Key Parameters"); st.write("**hidden_layer_sizes** : Number of neurons in hidden layers")
        
        st.subheader("🎮 Interactive Demo")
        dset = st.selectbox("Select Dataset", ["Iris", "Digits"], key="mlp_ds")
        if st.button("▶️ Run Demo", key="mlp_btn"):
            data = load_iris() if dset == "Iris" else load_digits()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = MinMaxScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                fig_cm = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"), x=data.target_names, y=data.target_names, text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.write("**Learning Curve**")
                train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
                fig_lc, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
                ax.set_title("Learning Curve"); ax.set_xlabel("Training examples"); ax.set_ylabel("Score"); ax.legend(loc="best")
                st.pyplot(fig_lc); plt.close(fig_lc)

    elif selected_algo == "Random Forest Classifier":
        st.subheader("Random Forest Classifier")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Description"); st.write("An ensemble method that constructs multiple decision trees using bagging and random feature subsets. Highly effective for identifying key health risk factors.")
            st.markdown("#### Advantages"); st.write("✅ Very high accuracy and robust to overfitting on noisy health data"); st.write("✅ Handles high-dimensional patient data effectively"); st.write("✅ Provides feature importance scores (e.g., which vital signs matter most)")
        with col2:
            st.markdown("#### Disadvantages"); st.write("⚠️ Slower to predict than single trees"); st.write("⚠️ Can be computationally intensive to train on massive EHRs")
        st.subheader("Implementation Example"); st.code("from sklearn.ensemble import RandomForestClassifier\n\nclf = RandomForestClassifier(n_estimators=50, max_depth=10)\nclf.fit(X_train, y_train)", language="python")
        st.subheader("Key Parameters"); st.write("**n_estimators** : Number of trees")
        
        st.subheader("🎮 Interactive Demo")
        dset = st.selectbox("Select Dataset", ["Iris", "Digits"], key="rf_ds")
        if st.button("▶️ Run Demo", key="rf_btn"):
            data = load_iris() if dset == "Iris" else load_digits()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = MinMaxScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            clf = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                fig_cm = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"), x=data.target_names, y=data.target_names, text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.write("**Learning Curve**")
                train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
                fig_lc, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
                ax.set_title("Learning Curve"); ax.set_xlabel("Training examples"); ax.set_ylabel("Score"); ax.legend(loc="best")
                st.pyplot(fig_lc); plt.close(fig_lc)

    elif selected_algo == "Extra Trees Classifier":
        st.subheader("Extra Trees Classifier")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Description"); st.write("Similar to Random Forest, but uses the entire training set and selects node splits completely at random. Fast and robust for multi-class health risk prediction.")
            st.markdown("#### Advantages"); st.write("✅ Computationally faster than Random Forest"); st.write("✅ Further reduces variance in noisy patient data"); st.write("✅ Robust to outliers")
        with col2:
            st.markdown("#### Disadvantages"); st.write("⚠️ May slightly underfit if health data is highly structured")
        st.subheader("Implementation Example"); st.code("from sklearn.ensemble import ExtraTreesClassifier\n\nclf = ExtraTreesClassifier(n_estimators=50)\nclf.fit(X_train, y_train)", language="python")
        st.subheader("Key Parameters"); st.write("**n_estimators** : Number of trees")
        
        st.subheader("🎮 Interactive Demo")
        dset = st.selectbox("Select Dataset", ["Iris", "Digits"], key="et_ds")
        if st.button("▶️ Run Demo", key="et_btn"):
            data = load_iris() if dset == "Iris" else load_digits()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = MinMaxScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            clf = ExtraTreesClassifier(n_estimators=50).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                fig_cm = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"), x=data.target_names, y=data.target_names, text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.write("**Learning Curve**")
                train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
                fig_lc, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
                ax.set_title("Learning Curve"); ax.set_xlabel("Training examples"); ax.set_ylabel("Score"); ax.legend(loc="best")
                st.pyplot(fig_lc); plt.close(fig_lc)

    elif selected_algo == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbors (KNN)")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Description"); st.write("Classifies a patient sample based on the majority risk label of its K closest patient neighbors in the clinical feature space.")
            st.markdown("#### Advantages"); st.write("✅ Simple and intuitive to understand for medical triage"); st.write("✅ No explicit training phase (lazy learner)")
        with col2:
            st.markdown("#### Disadvantages"); st.write("⚠️ Very slow prediction time for large patient databases"); st.write("⚠️ Highly sensitive to feature scaling (e.g., cholesterol vs. age)")
        st.subheader("Mathematical Formula (Euclidean Distance)"); st.latex(r"d(x, y) = \sqrt{\sum (x_i - y_i)^2}")
        st.subheader("Implementation Example"); st.code("from sklearn.neighbors import KNeighborsClassifier\n\nclf = KNeighborsClassifier(n_neighbors=5)\nclf.fit(X_train, y_train)", language="python")
        st.subheader("Key Parameters"); st.write("**n_neighbors** : Number of neighbors (K)")
        
        st.subheader("🎮 Interactive Demo")
        dset = st.selectbox("Select Dataset", ["Iris", "Digits"], key="knn_ds")
        if st.button("▶️ Run Demo", key="knn_btn"):
            data = load_iris() if dset == "Iris" else load_digits()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = MinMaxScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                fig_cm = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"), x=data.target_names, y=data.target_names, text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.write("**Learning Curve**")
                train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
                fig_lc, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
                ax.set_title("Learning Curve"); ax.set_xlabel("Training examples"); ax.set_ylabel("Score"); ax.legend(loc="best")
                st.pyplot(fig_lc); plt.close(fig_lc)

    elif selected_algo == "Ridge Classifier":
        st.subheader("Ridge Classifier")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Description"); st.write("Applies Ridge regression (L2 regularization) to classification by treating patient risk labels as +1 and -1. Great for avoiding overfitting.")
            st.markdown("#### Advantages"); st.write("✅ Extremely fast to train and predict on patient data"); st.write("✅ Handles multicollinearity between vital signs exceptionally well")
        with col2:
            st.markdown("#### Disadvantages"); st.write("⚠️ Assumes linear relationships between features and health outcomes")
        st.subheader("Implementation Example"); st.code("from sklearn.linear_model import RidgeClassifier\n\nclf = RidgeClassifier(alpha=1.0)\nclf.fit(X_train, y_train)", language="python")
        st.subheader("Key Parameters"); st.write("**alpha** : Regularization strength")
        
        st.subheader("🎮 Interactive Demo")
        dset = st.selectbox("Select Dataset", ["Iris", "Digits"], key="ridge_ds")
        if st.button("▶️ Run Demo", key="ridge_btn"):
            data = load_iris() if dset == "Iris" else load_digits()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = MinMaxScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            clf = RidgeClassifier(alpha=1.0).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                fig_cm = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"), x=data.target_names, y=data.target_names, text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.write("**Learning Curve**")
                train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
                fig_lc, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
                ax.set_title("Learning Curve"); ax.set_xlabel("Training examples"); ax.set_ylabel("Score"); ax.legend(loc="best")
                st.pyplot(fig_lc); plt.close(fig_lc)

    elif selected_algo == "Multinomial Naive Bayes":
        st.subheader("Multinomial Naive Bayes")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Description"); st.write("Designed for discrete medical records (frequency of symptoms or patient history). Assumes features follow a multinomial distribution.")
            st.markdown("#### Advantages"); st.write("✅ The go-to algorithm for text-based medical notes and symptom classification"); st.write("✅ Extremely fast and memory-efficient")
        with col2:
            st.markdown("#### Disadvantages"); st.write("⚠️ Strong independence assumption between medical findings"); st.write("⚠️ Requires discrete (count) medical history data to perform well")
        st.subheader("Implementation Example"); st.code("from sklearn.naive_bayes import MultinomialNB\n\nclf = MultinomialNB(alpha=1.0)\nclf.fit(X_train, y_train)", language="python")
        st.subheader("Key Parameters"); st.write("**alpha** : Additive (Laplace) smoothing parameter")
        
        st.subheader("🎮 Interactive Demo")
        dset = st.selectbox("Select Dataset", ["Iris", "Digits"], key="mnb_ds")
        if st.button("▶️ Run Demo", key="mnb_btn"):
            data = load_iris() if dset == "Iris" else load_digits()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = MinMaxScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            clf = MultinomialNB(alpha=1.0).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                fig_cm = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"), x=data.target_names, y=data.target_names, text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.write("**Learning Curve**")
                train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
                fig_lc, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
                ax.set_title("Learning Curve"); ax.set_xlabel("Training examples"); ax.set_ylabel("Score"); ax.legend(loc="best")
                st.pyplot(fig_lc); plt.close(fig_lc)

    elif selected_algo == "AdaBoost Classifier":
        st.subheader("AdaBoost Classifier")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Description"); st.write("An ensemble boosting algorithm that builds a sequence of weak learners focusing on previously misclassified patient samples.")
            st.markdown("#### Advantages"); st.write("✅ Highly accurate and reduces bias in health risk models effectively"); st.write("✅ Less prone to overfitting on medical data")
        with col2:
            st.markdown("#### Disadvantages"); st.write("⚠️ Highly sensitive to outliers and noisy patient records"); st.write("⚠️ Can be slow to train with many weak learners")
        st.subheader("Implementation Example"); st.code("from sklearn.ensemble import AdaBoostClassifier\n\nclf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)\nclf.fit(X_train, y_train)", language="python")
        st.subheader("Key Parameters"); st.write("**n_estimators** : Number of estimators"); st.write("**learning_rate** : Shrinks the contribution of each classifier")
        
        st.subheader("🎮 Interactive Demo")
        dset = st.selectbox("Select Dataset", ["Iris", "Digits"], key="ada_ds")
        if st.button("▶️ Run Demo", key="ada_btn"):
            data = load_iris() if dset == "Iris" else load_digits()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = MinMaxScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                fig_cm = px.imshow(confusion_matrix(y_test, y_pred), labels=dict(x="Predicted", y="Actual"), x=data.target_names, y=data.target_names, text_auto=True, color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                st.write("**Learning Curve**")
                train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
                fig_lc, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
                ax.set_title("Learning Curve"); ax.set_xlabel("Training examples"); ax.set_ylabel("Score"); ax.legend(loc="best")
                st.pyplot(fig_lc); plt.close(fig_lc)

# ==========================================
# PAGE 3: MODEL IMPLEMENTATION
# ==========================================
def implementation_page():
    st.title("⚙️ Model Implementation")
    
    col_upload, col_instructions = st.columns([1, 2])
    
    with col_upload:
        st.subheader("📤 Upload Model Files")
        model_file = st.file_uploader("Model File (.pkl)", type=['pkl'])
        scaler_file = st.file_uploader("Scaler File (.pkl)", type=['pkl'])
        uploaded = st.button("📤 Upload Files", type="primary")
        
        if uploaded and model_file is not None and scaler_file is not None:
            try:
                st.session_state['loaded_model'] = pickle.load(model_file)
                st.session_state['loaded_scaler'] = pickle.load(scaler_file)
                st.success("✅ Model and Scaler loaded successfully!")
            except Exception as e:
                st.error(f"Error loading files: {e}")
        elif uploaded:
            st.warning("Please upload both files.")

    with col_instructions:
        st.subheader("📝 Instructions:")
        st.info("1. Upload both model and scaler files from the sidebar.\n2. Enter feature values or use the randomize button.\n3. Click 'Make Prediction' to see the results.")

    if 'loaded_model' in st.session_state and 'loaded_scaler' in st.session_state:
        st.subheader("🔮 Make Prediction")
        
        features = st.text_input("Enter Features (comma-separated)", "25, 22.5, 120, 80, 180, 5.5")
        
        col_rand, col_pred = st.columns([1, 2])
        with col_rand:
            if st.button("🎲 Randomize Values"):
                rand_vals = np.random.randn(6).round(2) * 10 + 50
                features = ", ".join([str(v) for v in rand_vals])
                st.rerun()
        
        with col_pred:
            if st.button("🚀 Make Prediction", type="primary"):
                try:
                    vals = np.array([float(x.strip()) for x in features.split(",")]).reshape(1, -1)
                    scaler = st.session_state['loaded_scaler']
                    model = st.session_state['loaded_model']
                    
                    vals_scaled = scaler.transform(vals)
                    prediction = model.predict(vals_scaled)
                    
                    st.success(f"Predicted Class: **{prediction[0]}**")
                    
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(vals_scaled)[0]
                        prob_df = pd.DataFrame({'Class': model.classes_, 'Probability': probs})
                        st.bar_chart(prob_df, x="Class", y="Probability")
                except Exception as e:
                    st.error(f"Error making prediction: {e}. Ensure feature count matches the model.")

# ==========================================
# NAVIGATION SYSTEM
# ==========================================
pg = st.navigation([
    st.Page(generator_page, title="Home", icon="🏠"),
    st.Page(education_page, title="Algorithm Education", icon="📚"),
    st.Page(implementation_page, title="Model Implementation", icon="⚙️")
])

pg.run()

# Footer (shows on all pages)
st.markdown("---")
st.caption("Final project in Modeling and Simulation -Kathlene Laresma- BSCS 4A")
