selected_columns = ['Industry/Sector', 'Stage', 'Business Model', 'Loyal Customer']
data_selected = data_company[selected_columns]


label_encoders = {}
for col in ['Industry/Sector', 'Stage', 'Business Model']:
    le = LabelEncoder()
    data_selected[col] = le.fit_transform(data_selected[col])
    label_encoders[col] = le


scaler = MinMaxScaler()
data_selected['Loyal Customer'] = scaler.fit_transform(data_selected[['Loyal Customer']])


X = data_selected.values
target = (data_company['Loyal Customer'] > 5000).astype(int)  


X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)


model = models.Sequential([
    layers.Input(shape=(4,)),  
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


tflite_model_path = 'model_company_recommendation.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model saved to {tflite_model_path}")


def recommend_company(user_input):
    """
    Recommend a company based on user preferences.
    :param user_input: Dictionary containing user preferences
    """
   
    input_df = pd.DataFrame(user_input)

    
    for col in ['Industry/Sector', 'Stage', 'Business Model']:
        input_df[col] = label_encoders[col].transform(input_df[col])

    
    input_df['Loyal Customer'] = scaler.transform(input_df[['Loyal Customer']])

    
    model_input = input_df.values.astype('float32')

    
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    
    interpreter.set_tensor(input_details[0]['index'], model_input)

    
    interpreter.invoke()

    
    prediction = interpreter.get_tensor(output_details[0]['index'])

    if prediction > 0.5:
        recommended_companies = data_company[(data_company['Industry/Sector'].isin(user_input['Industry/Sector'])) &
                                            (data_company['Stage'].isin(user_input['Stage'])) &
                                            (data_company['Business Model'].isin(user_input['Business Model']))]
        return recommended_companies
    else:
        return "Consider other options based on your preferences."


user_input = {
    'Industry/Sector': ['Healthcare Technology'],
    'Stage': ['Series B'],
    'Business Model': ['Ad-based'],
    'Loyal Customer': [9325]
}


recommended_companies = recommend_company(user_input)
if isinstance(recommended_companies, pd.DataFrame):
    print("Recommended Companies based on your preferences:")
    print(recommended_companies[['Industry/Sector', 'Stage', 'Business Model', 'Loyal Customer']])
else:
    print(recommended_companies)
