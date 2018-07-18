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
        Bitmap test;
        public async Task Run()
        {
            evolver = new GeneticEvolver(layerRange: new Tuple<int, int>(3, 6),
                                                            layerLength: new Tuple<int, int>(3, 8),
                                                            evaluationFunc: this.Evaluate,
                                                            inputs: 30,
                                                            outputs: 2,
                                                            maxMutationFactor: 0.1,
                                                            mutationProbability: 0.3,
                                                            netPopulations: 10,
                                                            netsPerPopulation: 5,
                                                            activationFunction: System.Math.Tanh);
            evolver.Init();
            for (int i = 0; evolver.MaxFitness < 1; i++)
            {
                await evolver.Evaluate();
                Console.WriteLine("Generation {0} finished with max fitness {1}", evolver.Generation, evolver.MaxFitness);
                Console.Title = String.Format("Generation {0}; Max fitness {1}", evolver.Generation, evolver.MaxFitness);
                evolver.Evolve();
                await Task.Delay(100);
            }
        }

        void NewTest()
        {
            test = new Bitmap(8, 8);
            Graphics g = Graphics.FromImage(test);
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            int ch = RandomValues.RandomInt(48, 122);
            g.DrawString((char)ch + "", new Font("Arial", 6), Brushes.Black, new Rectangle(0, 0, 8, 8));
            test.Save("test.bmp");
            List<double> doubles = new List<double>();
            for (int x = 0; x < test.Width; x++)
                for (int y = 0; y < test.Height; y++)
                {
                    Color color = test.GetPixel(x, y);
                    double brightness = color.R + color.G + color.B;
                    brightness /= 3.0;
                    doubles.Add(brightness);
                }
            BitArray b = new BitArray(new byte[] { (byte)ch });
            int[] expected = b.Cast<bool>().Select(bit => bit ? 1 : 0).ToArray();
            double[] inputs = doubles.ToArray();
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
            double[] inputs = RandomDoubles();
            double[] outs = network.Calculate(inputs);
            double expected = inputs.Sum() > 1 ? 1 : 0;
            return 1 - Math.Abs(outs[0] - expected);
        }
    }
}
