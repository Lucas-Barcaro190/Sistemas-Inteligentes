#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_INPUTS 4
#define TOTAL_SAMPLES 150
#define NUM_TRIALS 10 // Number of times to run each combination to average results

// Enum and Perceptron struct
typedef enum { SETOSA, VERSICOLOR, VIRGINICA } IrisSpecies;
typedef struct {
    double *weights;
    double bias;
    double learningRate;
} Perceptron;

// A struct to hold the results of a single tuning run
typedef struct {
    double splitRatio;
    double learningRate;
    int epochs;
    double accuracy;
} TrialResult;

// Function prototypes
void initializePerceptron(Perceptron *p, double learningRate);
void freePerceptron(Perceptron *p);
int predict(Perceptron *p, double *inputs);
void train(Perceptron *p, double *inputs, int desiredOutput);
IrisSpecies speciesNameToEnum(char *species);
double runSingleTrial(int numEpochs, double learningRate, int numTrainingSamples, double trainingData[][NUM_INPUTS], int trainingOutputs[], int numTestingSamples, double testingData[][NUM_INPUTS], int testingOutputs[]);

int main() {
    srand(time(NULL));

    // --- 1. Define Expanded Hyperparameter Grids and Split Ratios ---
    double splitRatios[] = {0.9, 0.8, 0.7, 0.6, 0.5}; // 90/10, 80/20, etc.
    double learningRates[] = {0.5, 0.2, 0.1, 0.08, 0.05, 0.02, 0.01, 0.005};
    int epochOptions[] = {20, 50, 80, 100, 120, 150, 200, 300};
    int numSplitRatios = sizeof(splitRatios) / sizeof(splitRatios[0]);
    int numLearningRates = sizeof(learningRates) / sizeof(learningRates[0]);
    int numEpochOptions = sizeof(epochOptions) / sizeof(epochOptions[0]);

    // --- 2. Load and Prepare Data for "Versicolor vs. Virginica" ---
    double allData[TOTAL_SAMPLES][NUM_INPUTS];
    IrisSpecies allLabels[TOTAL_SAMPLES];
    FILE *file = fopen("iris.data", "r");
    if (file == NULL) { printf("Error: Could not open iris.data.\n"); return 1; }
    char line[100]; int count = 0;
    while (fgets(line, sizeof(line), file)) {
        char name[20];
        sscanf(line, "%lf,%lf,%lf,%lf,%s", &allData[count][0], &allData[count][1], &allData[count][2], &allData[count][3], name);
        allLabels[count++] = speciesNameToEnum(name);
    }
    fclose(file);

    double sessionData[100][NUM_INPUTS];
    int sessionLabels[100];
    int sessionCount = 0;
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        if (allLabels[i] == VERSICOLOR || allLabels[i] == VIRGINICA) {
            memcpy(sessionData[sessionCount], allData[i], NUM_INPUTS * sizeof(double));
            sessionLabels[sessionCount++] = (allLabels[i] == VIRGINICA);
        }
    }
    
    // --- 3. Run the Grid Search across all splits and hyperparameters ---
    int numCombinations = numSplitRatios * numLearningRates * numEpochOptions;
    TrialResult* allResults = (TrialResult*)malloc(numCombinations * NUM_TRIALS * sizeof(TrialResult));
    int resultIndex = 0;

    printf("Starting advanced hyperparameter tuning...\n");
    printf("Total combinations to test: %d. This will take some time.\n\n", numCombinations);

    // Outer loop for split ratios
    for (int s = 0; s < numSplitRatios; s++) {
        double currentRatio = splitRatios[s];
        int numTrainingSamples = (int)(sessionCount * currentRatio);
        int numTestingSamples = sessionCount - numTrainingSamples;
        
        printf("--- Testing Split Ratio: %d%% Train / %d%% Test ---\n", (int)(currentRatio*100), 100 - (int)(currentRatio*100));

        // Dynamically create data splits for the current ratio
        double trainingData[numTrainingSamples][NUM_INPUTS];
        int trainingOutputs[numTrainingSamples];
        double testingData[numTestingSamples][NUM_INPUTS];
        int testingOutputs[numTestingSamples];
        
        int trainIndex = 0, testIndex = 0;

        // CORRECTED: Using the dynamic interleaved split logic
        int cycleLength = 10;
        int trainThreshold = (int)(currentRatio * cycleLength);

        for (int i = 0; i < sessionCount; i++) {
            if (i % cycleLength < trainThreshold) {
                if (trainIndex < numTrainingSamples) {
                    memcpy(trainingData[trainIndex], sessionData[i], NUM_INPUTS * sizeof(double)); 
                    trainingOutputs[trainIndex++] = sessionLabels[i];
                }
            } else {
                if (testIndex < numTestingSamples) {
                    memcpy(testingData[testIndex], sessionData[i], NUM_INPUTS * sizeof(double)); 
                    testingOutputs[testIndex++] = sessionLabels[i];
                }
            }
        }

        // Inner loops for hyperparameter grid search
        for (int i = 0; i < numLearningRates; i++) {
            for (int j = 0; j < numEpochOptions; j++) {
                double totalAccuracy = 0;
                for (int k = 0; k < NUM_TRIALS; k++) {
                    double accuracy = runSingleTrial(epochOptions[j], learningRates[i], numTrainingSamples, trainingData, trainingOutputs, numTestingSamples, testingData, testingOutputs);
                    allResults[resultIndex].splitRatio = currentRatio;
                    allResults[resultIndex].learningRate = learningRates[i];
                    allResults[resultIndex].epochs = epochOptions[j];
                    allResults[resultIndex].accuracy = accuracy;
                    resultIndex++;
                    totalAccuracy += accuracy;
                }
            }
        }
    }

    // --- 4. Report and Export Results ---
    printf("\n--- Tuning Complete ---\n");

    FILE* resultsFile = fopen("advanced_hyperparameter_results.csv", "w");
    if (resultsFile) {
        fprintf(resultsFile, "SplitRatio,LearningRate,Epochs,Accuracy\n");
        for (int i = 0; i < resultIndex; i++) {
            fprintf(resultsFile, "%.1f,%.3f,%d,%.2f\n", allResults[i].splitRatio, allResults[i].learningRate, allResults[i].epochs, allResults[i].accuracy);
        }
        fclose(resultsFile);
        printf("Detailed trial results exported to advanced_hyperparameter_results.csv\n");
    }

    free(allResults);
    return 0;
}

// This function runs a single, complete training/testing cycle and returns the accuracy
double runSingleTrial(int numEpochs, double learningRate, int numTrainingSamples, double trainingData[][NUM_INPUTS], int trainingOutputs[], int numTestingSamples, double testingData[][NUM_INPUTS], int testingOutputs[]) {
    Perceptron p;
    initializePerceptron(&p, learningRate);

    for (int i = 0; i < numEpochs; i++) {
        for (int j = 0; j < numTrainingSamples; j++) {
            train(&p, trainingData[j], trainingOutputs[j]);
        }
    }
    int correct = 0;
    for (int i = 0; i < numTestingSamples; i++) {
        if (predict(&p, testingData[i]) == testingOutputs[i]) {
            correct++;
        }
    }
    freePerceptron(&p);
    return (double)correct / numTestingSamples * 100.0;
}

// --- Standard Helper and Perceptron Functions ---
IrisSpecies speciesNameToEnum(char *species) {
    if (strcmp(species, "Iris-setosa") == 0) return SETOSA;
    if (strcmp(species, "Iris-versicolor") == 0) return VERSICOLOR;
    if (strcmp(species, "Iris-virginica") == 0) return VIRGINICA;
    return -1;
}
void initializePerceptron(Perceptron *p, double learningRate) {
    p->weights = (double*)malloc(NUM_INPUTS * sizeof(double));
    p->learningRate = learningRate;
    for (int i = 0; i < NUM_INPUTS; i++) { p->weights[i] = ((double)rand() / RAND_MAX) - 0.5; }
    p->bias = ((double)rand() / RAND_MAX) - 0.5;
}
void freePerceptron(Perceptron *p) { free(p->weights); p->weights = NULL; }
int predict(Perceptron *p, double *inputs) {
    double sum = p->bias;
    for (int i = 0; i < NUM_INPUTS; i++) { sum += p->weights[i] * inputs[i]; }
    return sum >= 0 ? 1 : 0;
}
void train(Perceptron *p, double *inputs, int desired) {
    int prediction = predict(p, inputs);
    if (prediction != desired) {
        int error = desired - prediction;
        for (int i = 0; i < NUM_INPUTS; i++) { p->weights[i] += p->learningRate * error * inputs[i]; }
        p->bias += p->learningRate * error;
    }
}