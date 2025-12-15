import pandas as pd
import pulp as pl
import streamlit as st


def parse_csv(uploaded_file):
    """Parse gift data from CSV file."""
    df = pd.read_csv(uploaded_file)
    # Ensure proper column types
    df["kg"] = pd.to_numeric(df["kg"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    return df


def solve(df, max_kg, max_value):
    """Solve the gift packing optimization problem.

    Returns:
        tuple: (df_with_box, boxes_summary, num_boxes)
    """
    items = list(df.index)
    kg = df["kg"].to_dict()
    value = df["value"].to_dict()

    # Create decision variables
    num_boxes = len(items)
    B = list(range(num_boxes))
    x = pl.LpVariable.dicts("x", (items, B), 0, 1, pl.LpBinary)
    y = pl.LpVariable.dicts("y", B, 0, 1, pl.LpBinary)

    # Create optimization problem
    prob = pl.LpProblem("GiftPacking", pl.LpMinimize)

    # Objective: minimize number of boxes
    prob += pl.lpSum(y[b] for b in B)

    # Each item must be assigned to exactly one box
    for i in items:
        prob += pl.lpSum(x[i][b] for b in B) == 1

    # Weight and value constraints for each box
    for b in B:
        prob += pl.lpSum(kg[i] * x[i][b] for i in items) <= max_kg * y[b]
        prob += pl.lpSum(value[i] * x[i][b] for i in items) <= max_value * y[b]

    # Symmetry breaking constraint
    for b in range(len(B) - 1):
        prob += y[b] >= y[b + 1]

    # Solve
    prob.solve(pl.PULP_CBC_CMD(msg=False))

    if prob.status != pl.LpStatusOptimal:
        return None, None, 0

    # Extract results
    used_boxes = [b for b in B if pl.value(y[b]) > 0.5]

    # Create dataframe with box assignments
    df_result = df.copy()
    box_assignments = []
    for i in items:
        for b in B:
            if pl.value(x[i][b]) > 0.5:
                box_assignments.append(b + 1)
                break
    df_result["box"] = box_assignments

    # Create boxes summary
    boxes_summary = []
    for b in used_boxes:
        box_items = [i for i in items if pl.value(x[i][b]) > 0.5]
        w = sum(kg[i] for i in box_items)
        v = sum(value[i] for i in box_items)
        names = df.loc[box_items, "item"].tolist()
        boxes_summary.append({
            "Box": b + 1,
            "Weight (kg)": round(w, 2),
            "Value": round(v, 2),
            "Items": ", ".join(names)
        })

    return df_result, pd.DataFrame(boxes_summary), len(used_boxes)


def main():
    """Main function to orchestrate the gift packing optimization."""
    st.title("🎁 Gift Packing Optimizer")

    # Sidebar for constraints
    st.sidebar.header("Constraints")
    max_kg = st.sidebar.number_input("Max weight per box (kg)", value=2.0, step=0.1, min_value=0.1)
    max_value = st.sidebar.number_input("Max value per box (Currency)", value=39.0, step=1.0, min_value=1.0)

    st.write(f"Configure constraints in the sidebar. Current limits: {max_kg} kg and {max_value} currency per box.")

    uploaded_file = st.file_uploader("Upload gift data (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = parse_csv(uploaded_file)

        st.subheader("Gift Data (Editable)")
        edited_df = st.data_editor(df, num_rows="dynamic", width=stretch)

        if st.button("🎁 Pack Gifts"):
            if edited_df.empty:
                st.error("No gift data to optimize.")
            else:
                with st.spinner("Solving optimization problem..."):
                    df_result, boxes_summary, num_boxes = solve(edited_df.reset_index(drop=True), max_kg, max_value)

                    if df_result is None:
                        st.error("Could not find a feasible solution. Try relaxing constraints.")
                    else:
                        st.subheader("Results")
                        st.metric("Minimum boxes needed", num_boxes)

                        st.write("**Box Summary:**")
                        st.dataframe(boxes_summary, width=stretch)

                        st.write("**Item Assignments:**")
                        st.dataframe(df_result, width=stretch)

                        # Download button for results
                        csv = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="gift_packing_results.csv",
                            mime="text/csv"
                        )
    else:
        st.info("Please upload a CSV file with columns: 'item', 'kg', 'value'")


main()
