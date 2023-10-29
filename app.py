# Using Anaconda because default Python Don't have these libraries while Anaconda has
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

port = int(os.environ.get("PORT", 8501))
st.port = port


# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv', index_col=0)
data = pd.concat([train, test], axis=1)

houses_data = pd.read_csv('prediction_data_edited.csv')

keep_cols = ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'GarageArea', 
             'GrLivArea', 'LotArea', 'MasVnrArea', 'OpenPorchSF', 'OverallCond',
             'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd']

houses_data = houses_data[keep_cols]

st.set_option('deprecation.showPyplotGlobalUse', False)
# Load Model
model = joblib.load('xgb_r2.pkl')

# Page: Quick Project Summary
def quick_project_summary():
    st.markdown('Welcome to the House Price Prediction Dashboard! This comprehensive platform is your gateway to understanding and predicting house prices in Ames, Iowa. Whether you are a prospective homebuyer, a real estate enthusiast, or a data science enthusiast, this dashboard is tailored to provide you with a wealth of information and insights.')

    st.markdown('### **Project Overview:**')
    st.markdown('Harnessing the power of advanced machine learning techniques, this project endeavors to deliver accurate predictions for house prices. By analyzing an extensive range of features such as floor area, number of bedrooms, basement amenities, garage size, living area, and overall property condition, our model provides reliable estimates that empower you to make well-informed decisions in the real estate market.')

    st.markdown('### **Key Features of the Dashboard:**')
    st.markdown('- **Quick Project Summary:** Get an overview of the project objectives and methodology.')
    st.markdown('- **Sale Price Study:** Delve into a detailed exploration of correlations and visualizations. Understand how different features are interconnected and impact sale prices.')
    st.markdown('- **House Price Sale UI:** Input specific details of inherited houses, including dimensions, bedrooms, and more. Instantly receive accurate predictions for their potential sale prices.')
    st.markdown('- **ML Predict Sale Price:** Dive into the intricacies of our machine learning model. Explore interactive scatterplots that elucidate the influence of various features on predicted prices.')

    st.markdown('### **Navigating the Dashboard:**')
    st.markdown('1. **Quick Project Summary:** Begin your journey by gaining insights into the project objectives and methodologies.')
    st.markdown('2. **Sale Price Study:** Explore in-depth correlations, bar plots, scatter plots, and heatmaps. Understand the nuances of how different factors affect house prices.')
    st.markdown('3. **House Price Sale UI:** Provide specific details for inherited houses. Receive instant, data-driven predictions for potential sale prices based on your inputs.')
    st.markdown('4. **ML Predict Sale Price:** Explore interactive scatterplots to comprehend the relationships between features and predicted prices. Understand the model’s predictive power.')

    st.markdown('### **Why Trust Our Predictions:**')
    st.markdown('Our predictions are the result of meticulous data preprocessing, rigorous model training, and extensive validation. We have curated a high-quality dataset and fine-tuned our machine learning model to ensure accurate and reliable predictions. Trust in our insights to guide your real estate decisions.')

    st.markdown('### **Get Started:**')
    st.markdown('Begin your exploration of the dashboard by selecting a specific section from the navigation menu. Dive into the data, interact with the visualizations, and gain valuable insights into the intricate world of house pricing.')

    st.markdown('Thank you for choosing our House Price Prediction Dashboard. We invite you to explore, learn, and make informed decisions in the realm of real estate!')


# Page: Sale Price Study
def sale_price_study():
    st.title('Sale Price Study')
    st.header('Correlation Analysis')
    # Strongest Correlations Page
    st.header('Strongest Correlations with Sale Price')
    st.markdown('The following features have the strongest correlation with Sale Price:')
    strongest_correlations = data.corr()['SalePrice'].sort_values(ascending=False).head(5)
    st.write(strongest_correlations)
    # Heatmap
    st.subheader('Correlation Heatmap')
    corr = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    st.pyplot()


# Page: House Price Sale UI
def house_price_sale_ui():
    st.title('House Price Sale UI')
    # Inherited Houses Page
    st.header('Inherited Houses and Predicted Sale Prices')
    st.markdown('Provided details for 4 inherited houses predict their sale prices as follows.')
    if st.button('Predict Inherited Houses'):
        predictions = model.predict(houses_data)
        for i, prediction in enumerate(predictions):
            st.write(f'Inherited House {i + 1} Predicted Price: ${prediction:,.2f}')

        total_predicted_price = sum(predictions)
        st.write(f'Total Predicted Price for all 4 Inherited Houses: ${total_predicted_price:,.2f}')


    with st.sidebar:
        st.header('User Input for Price Prediction')
        user_first_floor_area = st.number_input('1st Floor Area (sq. ft.)', min_value=0)
        user_second_floor_area = st.number_input('2nd Floor Area (sq. ft.)', min_value=0)
        user_bedrooms = st.number_input('Number of Bedrooms', min_value=0)
        user_bsmt_finished_area = st.number_input('Basement Finished Area (sq. ft.)', min_value=0)
        user_garage_area = st.number_input('Garage Area (sq. ft.)', min_value=0)
        user_living_area = st.number_input('Living Area (sq. ft.)', min_value=0)
        user_lot_area = st.number_input('Lot Area (sq. ft.)', min_value=0)
        user_mas_vnr_area = st.number_input('Masonry Veneer Area', min_value=0)
        user_open_porch_area = st.number_input('Open Porch Area (sq. ft.)', min_value=0)
        user_overall_condition = st.slider('Overall Condition', min_value=1, max_value=10, step=1)
        user_overall_quality = st.slider('Overall Quality', min_value=1, max_value=10, step=1)
        user_total_bsmt_area = st.number_input('Total Basement Area (sq. ft.)', min_value=0)
        user_year_built = st.number_input('Year Built', min_value=0)
        user_year_remod_add = st.number_input('Year Remodeled', min_value=0)

        user_input = pd.DataFrame({
            '1stFlrSF': [user_first_floor_area],
            '2ndFlrSF': [user_second_floor_area],
            'BedroomAbvGr': [user_bedrooms],
            'BsmtFinSF1': [user_bsmt_finished_area],
            'GarageArea': [user_garage_area],
            'GrLivArea': [user_living_area],
            'LotArea': [user_lot_area],
            'MasVnrArea': [user_mas_vnr_area],
            'OpenPorchSF': [user_open_porch_area],
            'OverallCond': [user_overall_condition],
            'OverallQual': [user_overall_quality],
            'TotalBsmtSF': [user_total_bsmt_area],
            'YearBuilt': [user_year_built],
            'YearRemodAdd': [user_year_remod_add]
        })
    
    if st.button('Calculate User Input Price'):
        prediction = model.predict(user_input)
        st.subheader('Predicted House Price based on User Input:')
        st.write(f'${prediction[0]:,.2f}')



# Page: ML Predict Sale Price
def ml_predict_sale_price():
    st.title('ML Predict Sale Price')
    st.header('How ML Predicts Sale Price')
    
    st.markdown('Learn how the model predicts house prices using scatter plots and other visualizations.')

    # Model Explanation
    st.subheader('XGBoost Regression')
    st.markdown('We used the XGBoost regression algorithm to predict house prices. XGBoost is an advanced machine learning algorithm based on decision trees, which works well with structured data.')

    st.subheader('Model Performance')
    st.markdown('After applying various Feature Engineering techniques, the model achieved the following performance metrics:')
    st.markdown('- Mean Absolute Error (MAE): 15,213')
    st.markdown('- R-squared (R2) Score: 0.90 (90%) ')

    # Scatter Plot: GrLivArea vs SalePrice
    st.subheader('GrLivArea vs SalePrice')
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['GrLivArea'], y=data['SalePrice'])
    plt.xlabel('GrLivArea')
    plt.ylabel('SalePrice')
    st.pyplot()
    st.markdown("Here this graph shows that the GrLivArea is somewhat linear to SalePrice which we already saw in the Correlation page!")

# Main App
def main():
    st.sidebar.title('Navigation')
    page_options = ["Quick Project Summary", "Sale Price Study", "House Price Sale UI", "ML Predict Sale Price"]
    selected_page = st.sidebar.radio("Go to", page_options)

    if selected_page == "Quick Project Summary":
        quick_project_summary()
    elif selected_page == "Sale Price Study":
        sale_price_study()
    elif selected_page == "House Price Sale UI":
        house_price_sale_ui()
    elif selected_page == "ML Predict Sale Price":
        ml_predict_sale_price()

if __name__ == "__main__":
    main()
