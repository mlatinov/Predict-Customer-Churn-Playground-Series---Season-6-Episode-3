
from numpy import short
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ============= Additional Helpers =================================
def nn_data_split(data_transformed) :

    # Create a Validation set for NN from Dalex Transformed data 
    X_tr, X_val, y_tr, y_val = train_test_split(
    data_transformed["dalex_df"],
    data_transformed["y_encoded"],
    test_size    = 0.2,
    stratify     = data_transformed["y_encoded"],   
    random_state = 42
    )

    nn_data = {
        "x_train"      : X_tr,
        "x_validation" : X_val,
        "y_train"      : y_tr,
        "y_validation" : y_val
    }
    return nn_data

# ================== Deep Learning Blocks ================================
def dense_block(x, units, activation = "relu", dropout_rate = 0.3 ) :
    """
    Dense Layer -> BatchNormalization ->  Activation -> Dropout  
    """
    # Build the Dense Block 
    x = tf.keras.layers.Dense(units , use_bias= False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation = activation)(x)
    x = tf.keras.layers.Dropout(rate = dropout_rate)(x)
    return x 

def residual_block(x, units, activation = "relu", dropout_rate = 0.3, dense_layers_for_block = 2):
    """
    Residual block: the skip connection adds the input directly to the output
    """
    shortcut = x

    # Build Dense Block after that skip connection
    for i in dense_layers_for_block :
            x = tf.keras.layers.Dense(units , use_bias= False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation = activation)(x)
            x = tf.keras.layers.Dropout(rate = dropout_rate)(x)

    # Projection: match dimensions of shortcut to main path if needed
    if shortcut.shape[-1] != units:
        shortcut = tf.keras.layers.Dense(units, use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Add The Residual connection 
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation(activation = activation)(x)
    
    return x 

# ==================== Build Deep MLP =======================================
def build_deep_mlp(
    meta,
    hidden_units = [192, 128, 192, 224],
    dropout_rate = 0.15,
    activation = "elu",
    learning_rate = 0.0001
    ) :
    """
    Build Deep Multilayer Perception Network with Dense Block 
    """
    # Input Layer 
    n_features = meta["n_features_in_"]
    inputs = tf.keras.Input(shape=(n_features,), name="input_layer")
    x  = inputs

    # Loop over the elements in the hiidden units and build the network
    for units in hidden_units:
        x = dense_block(
            x            = x,
            units        = units,
            activation   = activation,
            dropout_rate = dropout_rate
        )

    # Output layer with Sigmoid activation function 
    output = tf.keras.layers.Dense(units = 1, activation = "sigmoid", name= "Output_Layer")(x)

    # Build the model 
    model = tf.keras.Model(inputs, output, name = "DeepMLPClassification")

    # Compile the model 
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = ["auc"]
    )
    return model 



