#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

using namespace std;

// Constants
const int BATCH_SIZE = 50;
const double HOEFFDING = 1.96;

// Utility function to generate random data
void generateSyntheticData(int numAtoms, int signalDim, int numSignals,
                           vector<vector<double>> &atoms,
                           vector<vector<double>> &signals,
                           vector<double> &varProxy,
                           vector<double> &maxmin)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    atoms.resize(numAtoms, vector<double>(signalDim));
    signals.resize(numSignals, vector<double>(signalDim));
    varProxy.resize(numSignals);

    double globalMax = -numeric_limits<double>::infinity();
    double globalMin = numeric_limits<double>::infinity();

    for (int i = 0; i < numAtoms; i++)
    {
        for (int j = 0; j < signalDim; j++)
        {
            atoms[i][j] = dis(gen);
        }
    }

    for (int i = 0; i < numSignals; i++)
    {
        double sum = 0.0, sumSquared = 0.0;
        for (int j = 0; j < signalDim; j++)
        {
            signals[i][j] = dis(gen);
            sum += signals[i][j];
            sumSquared += signals[i][j] * signals[i][j];
            globalMax = max(globalMax, signals[i][j]);
            globalMin = min(globalMin, signals[i][j]);
        }
        double mean = sum / signalDim;
        varProxy[i] = (sumSquared / signalDim) - (mean * mean);
    }

    maxmin = {globalMax, globalMin};
}

// Get confidence interval
double getCI(double delta, double varProxy, double ciBound, int numSamples)
{
    return ciBound * sqrt(varProxy / numSamples);
}

// Action Elimination function
pair<vector<vector<int>>, vector<int>> actionElimination(
    const vector<vector<double>> &atoms,
    const vector<vector<double>> &signals,
    const vector<double> &varProxy,
    const vector<double> &maxmin,
    double epsilon,
    double delta,
    bool verbose = false,
    int numBestAtoms = 2)
{

    int numSignals = signals.size();
    int numAtoms = atoms.size();
    int signalDim = atoms[0].size();

    vector<vector<int>> candidatesArray(numSignals, vector<int>(numBestAtoms));
    vector<int> budgetsArray(numSignals, 0);

    for (int signalIdx = 0; signalIdx < numSignals; signalIdx++)
    {
        const vector<double> &signal = signals[signalIdx];
        double localVarProxy = varProxy[signalIdx];

        vector<int> candidates(numAtoms);
        iota(candidates.begin(), candidates.end(), 0);

        vector<double> means(numAtoms, 0.0);
        vector<double> ucbs(numAtoms, numeric_limits<double>::infinity());
        vector<double> lcbs(numAtoms, -numeric_limits<double>::infinity());

        int t = 0;
        while (candidates.size() > numBestAtoms)
        {
            t++;
            int start = (t - 1) * BATCH_SIZE;
            int end = min(t * BATCH_SIZE, signalDim);

            for (int c : candidates)
            {
                double sum = 0.0;
                for (int i = start; i < end; i++)
                {
                    sum += atoms[c][i] * signal[i];
                }
                means[c] = (end * means[c] + sum) / end;
            }

            double ci = getCI(delta / numAtoms, localVarProxy, HOEFFDING, end);

            for (int c : candidates)
            {
                ucbs[c] = means[c] + ci;
                lcbs[c] = means[c] - ci;
            }

            auto compareUCB = [&](int a, int b)
            { return ucbs[a] > ucbs[b]; };
            sort(candidates.begin(), candidates.end(), compareUCB);

            double threshold = lcbs[candidates[numBestAtoms - 1]];
            candidates.erase(remove_if(candidates.begin(), candidates.end(), [&](int c)
                                       { return ucbs[c] < threshold; }),
                             candidates.end());
        }

        candidatesArray[signalIdx] = vector<int>(candidates.begin(), candidates.begin() + numBestAtoms);
        budgetsArray[signalIdx] = t * BATCH_SIZE;

        if (verbose)
        {
            cout << "Signal " << signalIdx << " completed in " << t << " iterations.\n";
        }
    }

    return {candidatesArray, budgetsArray};
}

// Main function
int main()
{
    int numAtoms = 100;
    int signalDim = 200;
    int numSignals = 10;

    vector<vector<double>> atoms;
    vector<vector<double>> signals;
    vector<double> varProxy;
    vector<double> maxmin;

    generateSyntheticData(numAtoms, signalDim, numSignals, atoms, signals, varProxy, maxmin);

    double epsilon = 0.1;
    double delta = 0.05;

    auto [candidatesArray, budgetsArray] = actionElimination(atoms, signals, varProxy, maxmin, epsilon, delta, true);

    for (int i = 0; i < candidatesArray.size(); i++)
    {
        cout << "Signal " << i << ": Best atoms: ";
        for (int c : candidatesArray[i])
        {
            cout << c << " ";
        }
        cout << "| Budgets used: " << budgetsArray[i] << endl;
    }

    return 0;
}
