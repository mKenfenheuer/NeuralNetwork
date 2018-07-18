using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Threading.Tasks;
using System.Collections;

using NeuralNet.Training.GeneticEvolve;
using NeuralNet.Network.Interfaces;
using NeuralNet.Math;

namespace NeuralNetTester
{
    class Program
    {
        static void Main(string[] args)
        {
            new Program().Run().Wait();
            Console.ReadKey();
        }

        GeneticEvolver evolver;
        int tests = 10;
        List<double[]> testInputs = new List<double[]>();
        public async Task Run()
        {

            evolver = new GeneticEvolver(layerRange: new Tuple<int, int>(3, 6),
                                                            layerLength: new Tuple<int, int>(3, 8),
                                                            evaluationFunc: this.Evaluate,
                                                            inputs: 5,
                                                            outputs: 2,
                                                            maxMutationFactor: 0.2,
                                                            mutationProbability: 0.5,
                                                            netPopulations: 10,
                                                            netsPerPopulation: 10,
                                                            activationFunction: System.Math.Tanh);

            for (int i = 0; i < tests; i++)
            {
                testInputs.Add(RandomDoubles());
                Console.WriteLine("Generation of test input {0}: [{1}]", i + 1, String.Join(",", testInputs[i]));
            }

            evolver.Init();

            while (evolver.MaxFitness < 1)
            {
                await evolver.Evaluate();
                Console.WriteLine("Generation {0} finished with max fitness {1}", evolver.Generation, evolver.MaxFitness);
                Console.Title = String.Format("Generation {0}; Max fitness {1}", evolver.Generation, evolver.MaxFitness);
                evolver.Evolve();
            }

            double[] inputs = testInputs[0];
            double output = evolver.BestNetwork.Calculate(inputs)[0];
            double expected = inputs.Sum() % 2;
            double fitness = Fitness(expected, output);
            Console.WriteLine("Best network calc: IN: [{0}] OUT:{1} EXPECT: {2} FIT: {3}", String.Join(",", inputs), Math.Round(output, 4), expected, fitness);
            Console.WriteLine("Best network passed with fitness {0}", Evaluate(evolver.BestNetwork));
        }

        double[] RandomDoubles()
        {
            double[] inp = new double[evolver.InputCount];
            for (int i = 0; i < inp.Length; i++)
                inp[i] = RandomValues.RandomDouble() > 0.5 ? 1 : 0;
            return inp;
        }

        double Evaluate(INeuralNet network)
        {
            double[] fitnessTests = new double[10];
            for (int i = 0; i < fitnessTests.Length; i++)
            {
                double[] inputs = testInputs[i];
                double[] outs = network.Calculate(inputs);
                double expected = inputs.Sum() > 1 ? 1 : 0;
                fitnessTests[i] = Fitness(outs[0], expected);
            }
            return fitnessTests.Average();
        }

        double Fitness(double expected, double real)
        {
            return 1 - Math.Abs(real - expected);
        }
    }
}
