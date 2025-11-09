# ============================================
# ADMIN DASHBOARD
# ============================================
elif role == "Admin":
    st.header("🧮 Admin Dashboard")
    df = fetch_all(task_index)
    leaves = fetch_all(leave_index)

    # --- TASK SUMMARY ---
    if not df.empty:
        st.subheader("📊 Task Summary Overview")
        df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
        df["marks"] = pd.to_numeric(df.get("marks", 0), errors="coerce").fillna(0)

        avg_completion = df["completion"].mean()
        avg_marks = df["marks"].mean()
        st.metric("Average Completion %", f"{avg_completion:.2f}")
        st.metric("Average Marks", f"{avg_marks:.2f}")

        st.dataframe(df[["employee", "department", "task", "completion", "marks", "status"]], use_container_width=True)

        # --- PERFORMANCE CLUSTERING ---
        st.subheader("🧩 Employee Performance Clustering")

        from sklearn.cluster import KMeans

        # Ensure enough records
        if len(df) >= 3:
            n_clusters = min(3, len(df))
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            df["cluster"] = km.fit_predict(df[["completion", "marks"]])

            # Create color-coded clusters
            fig = px.scatter(
                df,
                x="completion",
                y="marks",
                color=df["cluster"].astype(str),
                hover_data=["employee", "task", "department", "status"],
                title="Employee Clustering by Performance",
                labels={"completion": "Completion %", "marks": "Marks"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cluster summary
            st.markdown("### 📈 Cluster Summary")
            summary = df.groupby("cluster")[["completion", "marks"]].mean().reset_index()
            summary["Cluster Label"] = summary["cluster"].apply(
                lambda x: "⭐ Top Performers" if x == summary["marks"].idxmax()
                else ("⚙️ Average Performers" if x == summary["marks"].median() else "❗Low Performers")
            )
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("Not enough records for clustering analysis (need ≥ 3 records).")

    # --- LEAVE SUMMARY ---
    if not leaves.empty:
        st.subheader("📅 Leave Summary Overview")
        st.dataframe(leaves[["employee", "leave_type", "from_date", "to_date", "status"]], use_container_width=True)
        total = len(leaves)
        approved = len(leaves[leaves["status"] == "Approve"])
        pending = len(leaves[leaves["status"] == "Pending"])
        rejected = len(leaves[leaves["status"] == "Reject"])
        st.write(f"✅ Approved: {approved} | ⏳ Pending: {pending} | ❌ Rejected: {rejected} | 🧾 Total: {total}")
