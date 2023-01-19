
# Clean the data
def clean_up_data(data):
  # Hard copy and remove missing values
  clean_data = data.copy() 
  clean_data = clean_data.dropna()
  return clean_data

def standardise_data(data):
  
  scaled_data = data.copy()
  column_names = ["bill_length_mm",
                  "flipper_length_mm",
                  "bill_depth_mm",
                  "body_mass_g"]

  scaler = StandardScaler()
  scaled_data[column_names] = scaler.fit_transform(scaled_data[column_names]) 
  return scaled_data

# Create the input data
def create_input(data, columns_list):
  input_data = data[columns_list]
  return input_data


# Function to produce KMeans cluster model
def kmean_clusters(data, n_clusters):
  model = KMeans(n_clusters = n_clusters,
                 init = "random",
                 max_iter = 100,
                 random_state = 100)
  model.fit(data)

  return model

# Add new cluster labels to dataframe
def create_cluster_labels(data, model):
  data["cluster_label"] = model.labels_
  return data

# Create confusion matrix to compare clusters with the real species name
def create_confusion_matrix(prediction,actual):
  confusion_matrix = pd.crosstab(prediction,actual, normalize ="index")
  return confusion_matrix

# Printing the accuracy of the model
def print_accuracy(prediction,actual):
  confusion_matrix = create_confusion_matrix(prediction, actual)

  species_list = actual.unique()

  for species in species_list:
    accuracy = confusion_matrix[species].max()
    accuracy = round(accuracy * 100, 2)
    print(f"The accuracy for {species} is {accuracy} %")
  print("")

# Plot the bar charts of the labels vs species
def plot_confusion_matrix(prediction,actual):
  confusion_matrix = create_confusion_matrix(prediction, actual)
  confusion_matrix.plot.bar(rot=0)

# Running all the other functions in one step
def run_k_means(clean_data, columns_list, n_clusters):
  print(f"Using: {columns_list} to find {n_clusters} clusters.")
  input_data = create_input(clean_data,columns_list)
  model = kmean_clusters(input_data, n_clusters) # Make sure to use input_data
  clean_data = create_cluster_labels(clean_data, model)
  print_accuracy(clean_data["cluster_label"], clean_data["species"])

  return model

penguins_data = load_penguins()

run_k_means(penguins_data, ["bill_length_mm","bill_depth_mm"], 3)
run_k_means(penguins_data, ["bill_length_mm","bill_depth_mm","body_mass_g"], 3)
run_k_means(penguins_data, ["bill_length_mm","body_mass_g"], 3)
run_k_means(penguins_data, ["bill_length_mm","bill_depth_mm","flipper_length_mm"], 3)
