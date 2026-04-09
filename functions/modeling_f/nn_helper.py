
import tensorflow as tf
from sklearn.model_selection import train_test_split

def nn_data_split(data_transformed) :

    # Create a Validation set for NN from Dalex Transformed data 
    X_tr, X_val, y_tr, y_val = train_test_split(
    data_transformed["dalex_df"],
    data_transformed["y_encoded"],
    test_size= 0.2,
    stratify= data_transformed["y_encoded"],   
    random_state=42
    )

    nn_data = {
        "x_train"      : X_tr,
        "x_validation" : X_val,
        "y_train"      : y_tr,
        "y_validation" : y_val
    }
    return nn_data
    
def deep_mlp(x_train, y_train, x_validation, y_validation, units) :

    # Build Sequantial Deep Multilayer Perception Model 
    deep_mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(units = 256, activation="relu", name = "First_input_layer"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units = 128, activation="relu", name = "Second_Dense_Layer"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units = 64, activation="relu", name = "Thirth_Dense_Layer"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units= 1, activation ="sigmoid",name  = "Output_Sigmoid_Layer")
    ])
    # Compile the model 
    deep_mlp.compile(
        loss= tf.keras.losses.BinaryCrossentropy(),
        optimizer= tf.keras.optimizers.Adam(),
        metrics = ["auc"]
    )
    # Create a lr schedule
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epochs: 0.001 * 10**(epochs / 20))

    # Train the model 
    deep_mlp.fit(
        x = x_data,
        y = y_data,
        epochs=100,
        callbacks=[lr_schedule]
    )

    