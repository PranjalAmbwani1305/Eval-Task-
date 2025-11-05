with tab2:
    df = fetch_all()
    if df.empty:
        st.warning("No tasks found in Pinecone.")
    else:
        # Normalize column names and data types
        df.columns = [c.lower() for c in df.columns]
        df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
        df["reviewed"] = df.get("reviewed", False).astype(str).str.lower()

        # Show only tasks that are submitted but not reviewed
        pending_reviews = df[
            (df["completion"] >= 1) &  # task must be updated at least once
            (df["reviewed"] != "true")
        ]

        if pending_reviews.empty:
            st.info("No team-submitted tasks pending review.")
        else:
            st.subheader("Tasks Ready for Manager Review")
            for _, r in pending_reviews.iterrows():
                st.markdown(f"#### {r['task']}")
                st.write(f"**Employee:** {r.get('employee', 'Unknown')}")
                st.write(f"**Completion:** {r.get('completion', 0)}%")
                st.write(f"**Status:** {r.get('status', 'N/A')}")
                st.write(f"**Deadline Risk:** {r.get('deadline_risk', 'N/A')}")
                st.write(f"**Description:** {r.get('description', '')}")

                marks = st.number_input(f"Marks (0–5) for {r['task']}", 0.0, 5.0, step=0.1)
                comments = st.text_area(f"Manager Comments for {r['task']}", key=f"mgr_{r['_id']}")

                if st.button(f"Finalize Review for {r['task']}", key=f"btn_{r['_id']}"):
                    sent_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                    sentiment = "Positive" if sent_val == 1 else "Negative"
                    md = safe_meta({
                        **r,
                        "marks": marks,
                        "manager_comments": comments,
                        "reviewed": True,
                        "sentiment": sentiment,
                        "reviewed_on": now()
                    })
                    index.upsert([{"id": r["_id"], "values": rand_vec(), "metadata": md}])
                    st.success(f"Review completed for {r['task']} ({sentiment}).")
