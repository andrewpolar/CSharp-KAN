// ============================================================================
// KANKAN (Kolmogorov-Arnold Networks KAczmarz-Newton)
//
// Concept: Andrew Polar, Mike Poluektov
// Theory and algorithms are described in peer-reviewed publications.
// https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
// https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
// https://link.springer.com/article/10.1007/s10994-025-06800-6
// https://www.mdpi.com/2673-3951/6/3/88
// https://www.sciencedirect.com/science/article/pii/S0925231226021703
// Additional information, examples and documentation are available at
// OpenKAN.org.
//
// ---------------------------------------------------------------------------
// License
//
// If an end user somehow earns billions of US dollars using this software
// and later encounters the developer asking for spare change outside a
// McDonald's, the end user is under no obligation to buy the developer
// a sandwich.
//
// Symmetry Clause
//
// Likewise, if the developer becomes rich and famous by publishing this
// software and later meets an unfortunate end user who went bankrupt while
// using it, the developer is also under no obligation to buy the end user
// a sandwich.
//----------------------------------------------------------------------------

// This is C# version. Here is unit test for training to predict products of multiple  
// complex numbers. The target is vector.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

namespace ComplexD
{
    internal class Program
    {
        static void Validation(KAN kan, double[][] features, double[][] targets)
        {
            int nRecords = features.Length;
            int nFeatures = features[0].Length;
            double[] p = new double[2];
            double[] p1 = new double[nRecords];
            double[] p2 = new double[nRecords];
            double[] t1 = new double[nRecords];
            double[] t2 = new double[nRecords];
            for (int record = 0; record < nRecords; ++record)
            {
                kan.Predict(features[record]);
                p = kan.GetPrediction();
                p1[record] = p[0];
                p2[record] = p[1];
                t1[record] = targets[record][0];
                t2[record] = targets[record][1];
            }

            double pearson1 = Helper.Pearson(p1, t1);
            double pearson2 = Helper.Pearson(p2, t2);
            Console.WriteLine("Pearsons for validation: Re and Im {0:0.0000} {1:0.0000}", pearson1, pearson2);
        }

        static (double min, double max) FindLimits(double[][] targets)
        {
            int nRows = targets.GetLength(0);
            double min = targets[0][0];
            double max = targets[0][1];
            for (int i = 0; i < nRows; ++i)
            {
                if (targets[i][0] < min) min = targets[i][0];
                if (targets[i][0] > max) max = targets[i][0];
                if (targets[i][1] < min) min = targets[i][1];
                if (targets[i][1] > max) max = targets[i][1];
            }
            return (min, max);
        }

        static void Main(string[] args)
        {
            const int nTrainingRecords = 20000;
            const int nValidationRecords = 2000;
            const int nNumbers = 3;
            const int nFeatures = nNumbers * 2;
            const int nTargets = 2;
            const double featureMin = -1.0;
            const double featureMax = 1.0;
            //this is relative error for testing. when 0.0 the data is accurate.
            //the error is added to features before computing the target.
            const double errorRate = (featureMax - featureMin) * 0.0;
            const double alpha = 0.3;

            const int nDisjoints = 4;
            const int nDisjointSize = 6500;
            const int nLoops = 17;

            Random rng = new Random();
            double[][] features_training = new double[nTrainingRecords][];
            double[][] targets_training = new double[nTrainingRecords][];
            for (int record = 0; record < nTrainingRecords; ++record)
            {
                features_training[record] = Helper.GenerateComplexFeatures(rng, nNumbers, featureMin, featureMax);
                targets_training[record] = Helper.ComputeComplexTargets(features_training[record], errorRate, rng);
            }

            double[][] features_validation = new double[nValidationRecords][];
            double[][] targets_validation = new double[nValidationRecords][];
            for (int record = 0; record < nValidationRecords; ++record)
            {
                features_validation[record] = Helper.GenerateComplexFeatures(rng, nNumbers, featureMin, featureMax);
                targets_validation[record] = Helper.ComputeComplexTargets(features_validation[record], errorRate, rng);
            }

            (double targetMin, double targetMax) = FindLimits(targets_training);
   
            Stopwatch sw = Stopwatch.StartNew();

            //this is network configuration
            List<int> network = new List<int>() { nFeatures, 40, nTargets };
            List<int> points = new List<int> { 4, 20 };

            KAN[] kans = new KAN[nDisjoints];

            kans[0] = new KAN(network, points, featureMin, featureMax, targetMin,
                targetMax, targetMin, targetMax, alpha, rng);

            for (int m = 1; m < nDisjoints; ++m)
            {
                kans[m] = new KAN(kans[0]);
            }

            for (int loop = 0; loop < nLoops; ++loop)
            {
                Parallel.For(0, nDisjoints, m =>
                {
                    int start = ((loop * nDisjoints + m) * nDisjointSize) % nTrainingRecords;
                    for (int i = 0; i < nDisjointSize; ++i)
                    {
                        int idx = (start + i) % nTrainingRecords;
                        kans[m].Forward(features_training[idx]);
                        kans[m].Update(targets_training[idx]);
                    }
                });

                // Merge: accumulate all disjoints into kans[0]
                for (int m = 1; m < nDisjoints; ++m)
                {
                    kans[0].Accumulate(kans[m]);
                }

                // Average
                kans[0].Scale(1.0 / nDisjoints);

                // Copy averaged KAN back to all disjoints
                for (int m = 1; m < nDisjoints; ++m)
                {
                    kans[m] = new KAN(kans[0]);
                }

                Console.WriteLine($"Loop {loop + 1} of {nLoops}");
            }

            Validation(kans[0], features_validation, targets_validation);
            sw.Stop();
            Console.WriteLine($"Time: {sw.Elapsed.TotalMilliseconds:F3} ms");
        }
    }
}

