#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_EPOCHS 200
#define LEARNING_RATE 0.03
#define TOTAL_SAMPLES 150

// Enum for species
typedef enum { SETOSA, VERSICOLOR, VIRGINICA } IrisSpecies;

// --- Data Structures for the MLP ---
typedef struct {
    double *weights;
    double bias;
    double output;
    double delta;
    int numInputs;
} Neuron;

typedef struct {
    Neuron *neurons;
    int numNeurons;
} Layer;

typedef struct {
    Layer *layers;
    int numLayers;
} MLP;

// --- Function Prototypes ---
MLP* createMLP(int numLayers, const int neuronsPerLayer[]);
void freeMLP(MLP *net);
void forwardPass(MLP *net, double inputs[]);
void backwardPass(MLP *net, double targets[]);
void updateWeights(MLP *net, double inputs[]);
void shuffleData(int n, double data[][4], double targets[][3]);
double calculateAccuracy(MLP *net, int numSamples, double data[][4], double targets[][3]);
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoidDerivative(double x) { return x * (1.0 - x); }
IrisSpecies speciesNameToEnum(char *species);

int main() {
    srand(time(NULL));

    // --- 1. Load Full 4D Data ---
    double allData[TOTAL_SAMPLES][4];
    IrisSpecies allLabels[TOTAL_SAMPLES];
    FILE *file = fopen("iris.data", "r");
    if (!file) {
        printf("Error: iris.data not found. Make sure it's in the same folder.\n");
        return 1;
    }
    char line[100];
    int count = 0;
    while (fgets(line, sizeof(line), file) && count < TOTAL_SAMPLES) {
        char name[20];
        sscanf(line, "%lf,%lf,%lf,%lf,%s", &allData[count][0], &allData[count][1], &allData[count][2], &allData[count][3], name);
        allLabels[count++] = speciesNameToEnum(name);
    }
    fclose(file);

    double allTargets[TOTAL_SAMPLES][3] = {{0}};
    for(int i=0; i<TOTAL_SAMPLES; ++i) {
        allTargets[i][allLabels[i]] = 1.0;
    }

    // --- 2. Shuffle the Dataset ---
    shuffleData(TOTAL_SAMPLES, allData, allTargets);
    printf("Dataset shuffled randomly.\n");

    // --- 3. Split Data into Training and Validation Sets ---
    int numTrainingSamples = 120; // 80% of 100
    int numValidationSamples = 30; // 20% of 100
    
    // The first 80 samples of the shuffled data are for training
    double (*trainingData)[4] = allData;
    double (*trainingTargets)[3] = allTargets;
    
    // The next 20 samples are for validation
    double (*validationData)[4] = &allData[numTrainingSamples];
    double (*validationTargets)[3] = &allTargets[numTrainingSamples];

    printf("Data split into %d training samples and %d validation samples.\n", numTrainingSamples, numValidationSamples);

    // --- 4. Create MLP and Output File ---
    const int architecture[] = {4, 8, 3};
    MLP *net = createMLP(3, architecture);
    FILE *weightsFile = fopen("mlp_weights_4d.txt", "w");
    if (!weightsFile) {
        printf("Error creating weights file.\n");
        return 1;
    }
    
    printf("Training 4D MLP and exporting weights for %d epochs...\n", MAX_EPOCHS);
    
    // --- 5. Train and Export Weights Every Epoch ---
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        // MODIFIED: Train ONLY on the training subset
        for (int i = 0; i < numTrainingSamples; i++) {
            forwardPass(net, trainingData[i]);
            backwardPass(net, trainingTargets[i]);
            updateWeights(net, trainingData[i]);
        }
        
        // (Optional but good practice) Print validation accuracy to monitor progress
        if ((epoch + 1) % 10 == 0) {
            double accuracy = calculateAccuracy(net, numValidationSamples, validationData, validationTargets);
            printf("Epoch %d/%d - Validation Accuracy: %.2f%%\n", epoch + 1, MAX_EPOCHS, accuracy);
        }

        // Export weights for this epoch (this logic is unchanged)
        fprintf(weightsFile, "EPOCH:%d\n", epoch + 1);
        for (int i = 0; i < net->numLayers; i++) {
            fprintf(weightsFile, "LAYER:%d\n", i + 1);
            for (int j = 0; j < net->layers[i].numNeurons; j++) {
                fprintf(weightsFile, "NEURON:%d", j + 1);
                for (int k = 0; k < net->layers[i].neurons[j].numInputs; k++) {
                    fprintf(weightsFile, ":%.8f", net->layers[i].neurons[j].weights[k]);
                }
                fprintf(weightsFile, ":%.8f\n", net->layers[i].neurons[j].bias);
            }
        }
    }

    fclose(weightsFile);
    freeMLP(net);
    printf("Weight export complete to mlp_weights_4d.txt\n");
    return 0;
}

// --- NEW SHUFFLE FUNCTION ---
void shuffleData(int n, double data[][4], double targets[][3]) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1); // Pick a random index from 0 to i

        // Swap data row
        double tempData[4];
        memcpy(tempData, data[i], 4 * sizeof(double));
        memcpy(data[i], data[j], 4 * sizeof(double));
        memcpy(data[j], tempData, 4 * sizeof(double));

        // Swap target row
        double tempTarget[3];
        memcpy(tempTarget, targets[i], 3 * sizeof(double));
        memcpy(targets[i], targets[j], 3 * sizeof(double));
        memcpy(targets[j], tempTarget, 3 * sizeof(double));
    }
}

// --- NEW ACCURACY CALCULATION FUNCTION ---
double calculateAccuracy(MLP *net, int numSamples, double data[][4], double targets[][3]) {
    int correct = 0;
    for (int i = 0; i < numSamples; i++) {
        forwardPass(net, data[i]);
        Layer *outputLayer = &net->layers[net->numLayers - 1];
        
        int predictedClass = 0;
        for(int j = 1; j < outputLayer->numNeurons; j++) {
            if(outputLayer->neurons[j].output > outputLayer->neurons[predictedClass].output) {
                predictedClass = j;
            }
        }
        
        int trueClass = 0;
        for(int j = 1; j < 3; j++) {
            if(targets[i][j] > targets[i][trueClass]) {
                trueClass = j;
            }
        }

        if (predictedClass == trueClass) {
            correct++;
        }
    }
    return (double)correct / numSamples * 100.0;
}

// --- Core MLP Implementation ---

MLP* createMLP(int numLayers, const int neuronsPerLayer[]) {
    MLP *net = (MLP*)malloc(sizeof(MLP));
    net->numLayers = numLayers - 1;
    net->layers = (Layer*)malloc(net->numLayers * sizeof(Layer));

    for (int i = 0; i < net->numLayers; i++) {
        net->layers[i].numNeurons = neuronsPerLayer[i + 1];
        net->layers[i].neurons = (Neuron*)malloc(net->layers[i].numNeurons * sizeof(Neuron));
        int numInputsForLayer = neuronsPerLayer[i];
        for (int j = 0; j < net->layers[i].numNeurons; j++) {
            net->layers[i].neurons[j].numInputs = numInputsForLayer;
            net->layers[i].neurons[j].weights = (double*)malloc(numInputsForLayer * sizeof(double));
            for (int k = 0; k < numInputsForLayer; k++) {
                net->layers[i].neurons[j].weights[k] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
            net->layers[i].neurons[j].bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    return net;
}

void freeMLP(MLP *net) {
    for (int i = 0; i < net->numLayers; i++) {
        for (int j = 0; j < net->layers[i].numNeurons; j++) {
            free(net->layers[i].neurons[j].weights);
        }
        free(net->layers[i].neurons);
    }
    free(net->layers);
    free(net);
}

void forwardPass(MLP *net, double inputs[]) {
    // First hidden layer
    Layer *firstLayer = &net->layers[0];
    for (int j = 0; j < firstLayer->numNeurons; j++) {
        double activation = firstLayer->neurons[j].bias;
        for (int k = 0; k < firstLayer->neurons[j].numInputs; k++) {
            activation += firstLayer->neurons[j].weights[k] * inputs[k];
        }
        firstLayer->neurons[j].output = sigmoid(activation);
    }

    // Subsequent layers
    for (int i = 1; i < net->numLayers; i++) {
        Layer *prevLayer = &net->layers[i - 1];
        Layer *currentLayer = &net->layers[i];
        for (int j = 0; j < currentLayer->numNeurons; j++) {
            double activation = currentLayer->neurons[j].bias;
            for (int k = 0; k < prevLayer->numNeurons; k++) {
                activation += currentLayer->neurons[j].weights[k] * prevLayer->neurons[k].output;
            }
            currentLayer->neurons[j].output = sigmoid(activation);
        }
    }
}

void backwardPass(MLP *net, double targets[]) {
    // Deltas for the output layer
    Layer *outputLayer = &net->layers[net->numLayers - 1];
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        double error = targets[i] - outputLayer->neurons[i].output;
        outputLayer->neurons[i].delta = error * sigmoidDerivative(outputLayer->neurons[i].output);
    }

    // Deltas for hidden layers
    for (int i = net->numLayers - 2; i >= 0; i--) {
        Layer *hiddenLayer = &net->layers[i];
        Layer *nextLayer = &net->layers[i + 1];
        for (int j = 0; j < hiddenLayer->numNeurons; j++) {
            double error = 0.0;
            for (int k = 0; k < nextLayer->numNeurons; k++) {
                error += nextLayer->neurons[k].weights[j] * nextLayer->neurons[k].delta;
            }
            hiddenLayer->neurons[j].delta = error * sigmoidDerivative(hiddenLayer->neurons[j].output);
        }
    }
}

void updateWeights(MLP *net, double inputs[]) {
    for (int i = 0; i < net->numLayers; i++) {
        Layer *prevLayer = (i > 0) ? &net->layers[i - 1] : NULL;
        for (int j = 0; j < net->layers[i].numNeurons; j++) {
            for (int k = 0; k < net->layers[i].neurons[j].numInputs; k++) {
                double inputVal = (i == 0) ? inputs[k] : prevLayer->neurons[k].output;
                net->layers[i].neurons[j].weights[k] += LEARNING_RATE * net->layers[i].neurons[j].delta * inputVal;
            }
            net->layers[i].neurons[j].bias += LEARNING_RATE * net->layers[i].neurons[j].delta;
        }
    }
}

IrisSpecies speciesNameToEnum(char *species) {
    if (strcmp(species, "Iris-setosa") == 0) return SETOSA;
    if (strcmp(species, "Iris-versicolor") == 0) return VERSICOLOR;
    if (strcmp(species, "Iris-virginica") == 0) return VIRGINICA;
    return -1;
}