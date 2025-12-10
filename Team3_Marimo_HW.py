# /// script
# dependencies = ["marimo", "and", "uv"]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App()


@app.cell
def _():
    # packages added via marimo's package management: marimo and uv !pip install marimo and uv
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import RandomForestRegressor
    return (
        RandomForestRegressor,
        cross_val_predict,
        cross_val_score,
        mo,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Delivery time delay predicting dashboard
    """)
    return


@app.cell
def _(pd):
    # uploading files from the dataset
    feedback_df = pd.read_csv(r'archive (1)\blinkit_customer_feedback.csv')
    orders_df = pd.read_csv(r'archive (1)\blinkit_orders.csv')
    delivery_df = pd.read_csv(r'archive (1)\blinkit_delivery_performance.csv')
    order_items_df = pd.read_csv(r'archive (1)\blinkit_order_items.csv')
    customers_df = pd.read_csv(r'archive (1)\blinkit_customers.csv')
    products_df = pd.read_csv(r'archive (1)\blinkit_products.csv')
    inventory_df = pd.read_csv(r'archive (1)\blinkit_inventoryNew.csv') 
    marketing_df = pd.read_csv(r'archive (1)\blinkit_marketing_performance.csv')
    return customers_df, delivery_df, feedback_df, order_items_df, orders_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Merging imported data to make one general table
    """)
    return


@app.cell
def _(customers_df, delivery_df, feedback_df, order_items_df, orders_df, pd):
    # merging imported data to make one general table

    # aggregating quantity of products in orders
    order_aggregates = order_items_df.groupby('order_id').agg(
        total_items=('quantity', 'sum'),      # total quantity of products in order
        unique_items=('product_id', 'nunique') # quantity of unique products in order
    ).reset_index()

    # merging tables step by step 
    # (only those tables, which content 1 line for 1 order, so the final data will be applicable for calculations)
    merged_df = pd.merge(orders_df, delivery_df, on='order_id', how='inner')
    merged_df = pd.merge(merged_df, customers_df, on='customer_id', how='inner')
    merged_df = pd.merge(merged_df, order_aggregates, on='order_id', how='inner')
    merged_df = pd.merge(merged_df, feedback_df[['order_id', 'rating']], on='order_id', how='left')

    # calculating new features based on the initial data
    ### converting the data types of the needed features to datetime
    merged_df['order_date'] = pd.to_datetime(merged_df['order_date'])
    merged_df['registration_date'] = pd.to_datetime(merged_df['registration_date'])
    merged_df['promised_delivery_time'] = pd.to_datetime(merged_df['promised_delivery_time'])
    ### calculating new features: order_hour, days_as_customer
    merged_df['order_hour'] = merged_df['order_date'].dt.hour
    merged_df['days_as_customer'] = (merged_df['order_date'] - merged_df['registration_date']).dt.days
    merged_df['promised_delivery_time_minutes'] = (merged_df['promised_delivery_time'] - merged_df['order_date']).dt.total_seconds() / 60

    # deleting duplicates if they occured after merging the tables from the dataset
    merged_df.drop_duplicates(subset=['order_id'], inplace=True)

    # deleting columns that duplicated after the merge:
    merged_df.drop(columns=['delivery_partner_id_y', 'delivery_status_y'], inplace=True)

    # renaming columns that were duplicated
    merged_df.rename(columns={'delivery_partner_id_x': 'delivery_partner_id', 'delivery_status_x': 'delivery_status'}, inplace=True)

    # print("Merged_df data structure:")
    # merged_df.info()
    merged_df
    return (merged_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Choose features for Model of Delivery time delay
    """)
    return


@app.cell
def _(merged_df, mo):
    # Creating a list with all possible features, excluding the Target and other features, that cannot be used in model
    all_possible_features = [
        col for col in merged_df.columns if col not in [
            'order_id', 'customer_id', 'delivery_partner_id', 'store_id', 
            'order_date', 'registration_date', 'promised_delivery_time', 
            'actual_delivery_time', 'actual_time', 'reasons_if_delayed', 
            'pincode', 'customer_name', 'email', 'phone', 'address', 'area', 
            'promised_time',
            'delivery_time_minutes' # target
        ]
    ]

    default_set_of_features = ['order_total', 'payment_method', 'distance_km', 'customer_segment', 'total_orders', 'avg_order_value', 'total_items', 'unique_items', 'order_hour', 'days_as_customer', 'promised_delivery_time_minutes']

    # Creating interactive element for features choice
    feature_selector = mo.ui.multiselect(
        options=all_possible_features,
        value=default_set_of_features, # choosing features for default set
        label="Available features"
    )
    # print('Available features:', *all_possible_features,  sep = '\n -')
    # Showing the interactive element for users
    feature_selector
    return (feature_selector,)


@app.cell
def _(feature_selector, merged_df, mo, pd):
    # Recieving chosen by user features:
    selected_features = feature_selector.value
    target = 'delivery_time_minutes'

    # Creating an error comment for cases, when user did not choose any feature
    if not selected_features:
        mo.stop(True, mo.md("**Error: Choose at least one feature to teach model on.**"))

    model_df = merged_df[selected_features + [target]].copy().dropna()

    # clarifying which of the selected features are categorical
    categorical_cols_to_encode = [
        col for col in ['customer_segment', 'payment_method', 'delivery_status'] 
        if col in selected_features
    ]

    # applying one-hot encoding only to selected categorical features:
    if categorical_cols_to_encode:
        model_df = pd.get_dummies(model_df, columns=categorical_cols_to_encode, drop_first=True)

    X = model_df.drop(columns=[target])
    y = model_df[target]

    mo.md(f"Data is prepared. **{len(X.columns)}** features will be used for training.")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choose the parameters of the RandomForestRegressor model:
    """)
    return


@app.cell
def _(mo):
    # Creating sliders to choose the parameters of the RandomForestRegressor model
    n_estimators_slider = mo.ui.slider(10, 500, step=10, value=100, label="Quantity of trees (n_estimators)")
    max_depth_slider = mo.ui.slider(2, 30, value=10, label="Max. tree depth (max_depth)")
    # random_state_slider = mo.ui.slider(0, 100, value=42, label="Состояние случайности (random_state)")
    # n_jobs_slider = mo.ui.slider(-1, 8, value=-1, label="Количество паралл. процессов (n_jobs)")

    # displaying sliders in the interface
    mo.hstack([n_estimators_slider, max_depth_slider], justify='space-around')
    return max_depth_slider, n_estimators_slider


@app.cell
def _(RandomForestRegressor, X, max_depth_slider, n_estimators_slider, y):
    # Getting information from sliders
    n_estimators = n_estimators_slider.value
    max_depth = max_depth_slider.value

    # Creating and teaching a model with chosen parameters on training data
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42, 
        n_jobs=-1       
    )

    model.fit(X, y)
    return (model,)


@app.cell
def _(X, cross_val_score, mo, model, np, y):
    # Evaluating the quality of the model by cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    avg_mae = -np.mean(cv_scores)
    mo.md(f"The model is evaluated. Average error (MAE): **{avg_mae:.2f} minutes**.")
    return (avg_mae,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Set the order parameters:
    """)
    return


@app.cell
def _(feature_selector, merged_df, mo, pd):
    # Getting an relevant list of features selected for the model
    selected_features_for_ui = feature_selector.value

    # Сreating a dictionary where all UI elements will be stored.
    # Key: name of the feature, Value: widget itself (slider/dropdown)
    prediction_inputs = {}

    # Walking through each selected feature and creating a widget for it.
    for feature in selected_features_for_ui:
        # If the attribute is numeric, create a slider.
        if pd.api.types.is_numeric_dtype(merged_df[feature]):
            min_val = float(merged_df[feature].min())
            max_val = float(merged_df[feature].max())
            median_val = float(merged_df[feature].median())

            # Defining the step for the slider
            step = 1 if pd.api.types.is_integer_dtype(merged_df[feature].dtype) else (max_val - min_val) / 100

            prediction_inputs[feature] = mo.ui.slider(
                start=min_val, stop=max_val, value=median_val, label=str(feature), step=step
            )
        # If the attribute is categorical (text), create a popup list.
        else:
            options = merged_df[feature].dropna().unique().tolist()
            prediction_inputs[feature] = mo.ui.dropdown(
                options=options, value=options[0], label=str(feature)
            )

    # In the Dataset of Blinkit Shop only one product in each order,  
    # therefore only one number on the slider of unique items (=1).

    # Also, every order consists of 1 to 3 units of product, that's why on slider of 
    # total items only values 1, 2, and 3

    # Displaying all created widgets
    mo.vstack(list(prediction_inputs.values()))
    return (prediction_inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forecast of Delivery_time_minutes:
    """)
    return


@app.cell
def _(X, avg_mae, merged_df, mo, model, pd, prediction_inputs):
    # Starting with an average values for all features in the trained model
    input_data = X.mean().to_dict()

    # Updating values based on user selection in the dynamic widgets
    for feature_, widget in prediction_inputs.items():
        widget_value = widget.value

        # If it's a numeric feature, simply update its value
        if pd.api.types.is_numeric_dtype(merged_df[feature_]):
            if feature_ in input_data:
                input_data[feature_] = widget_value

        # If it's a categorical feature, we need to handle one-hot encoding
        else:
            # First, reset all related one-hot columns to 0
            for col in X.columns:
                if col.startswith(f"{feature_}_"):
                    input_data[col] = 0
            # Then, set 1 for the selected value
            selected_column_name = f"{feature_}_{widget_value}"
            if selected_column_name in input_data:
                input_data[selected_column_name] = 1

    # Creating a DataFrame from the single row and making a prediction
    input_df = pd.DataFrame([input_data])
    predicted_time = model.predict(input_df)[0]

    # Displaying the result
    mo.stat(
        label="Delivery time delay forecast",
        value=f"{predicted_time:.0f} minutes",
        caption=f"Quality of the current model (MAE): {avg_mae:.2f} min."
    )
    return


@app.cell
def _(X, cross_val_predict, model, np, plt, y):
    # Visualization of real versus predicted values
    cv_predictions = cross_val_predict(model, X, y, cv=5, n_jobs=-1)

    def plot_predictions(y_true, y_pred):
        # A function to create the Real vs. Predicted scatter plot
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.scatter(y_true, y_pred, alpha=0.5, label='Forecasts')

        # Adding the diagonal line for reference
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal forecast')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Real delivery time (minutes)')
        ax.set_ylabel('Predicted delivery time (minutes)')
        ax.set_title('Real vs. Predicted values')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        return fig

    # Calling the function to display the plot
    plot_predictions(y, cv_predictions)
    return


@app.cell
def _(X, model, pd, plt):
    # Creating a plot of Importance of features for the Model
    def plot_feature_importance(model_to_plot, feature_names):
        importances = model_to_plot.feature_importances_
        forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        forest_importances.sort_values(ascending=True).plot.barh(ax=ax)
        ax.set_title("Importance of features for the Model")
        ax.set_xlabel("Mean decrease in impurity")
        fig.tight_layout()
        return fig

    plot_feature_importance(model, X.columns)
    return


if __name__ == "__main__":
    app.run()
