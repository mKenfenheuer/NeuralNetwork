using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Collections;
using System.Threading.Tasks;


using NeuralNet.Training.GeneticEvolve;
using NeuralNet.Network.Interfaces;
using NeuralNet.Math;
namespace NeuralNetTester
{
    class OCRTest
    {
        GeneticEvolver evolver;
        List<double[]> testInputs = new List<double[]>();
        List<double[]> testOutputs = new List<double[]>();
        public async Task<int> Run()
        {
            GenerateTestInputs();
            evolver = new GeneticEvolver(layerRange: new Tuple<int, int>(Math.Min(testInputs[0].Length, testOutputs[0].Length), Math.Max(testInputs[0].Length, testOutputs[0].Length)),
                                                            layerLength: new Tuple<int, int>(3, 5),
                                                            evaluationFunc: this.Evaluate,
                                                            inputs: testInputs[0].Length,
                                                            outputs: testOutputs[0].Length,
                                                            activationFunction: ActivationFunc,
                                                            mutationFunction: MutationFunc);
            evolver.Init();
            while (evolver.MaxFitness < 1)
            {
                await evolver.Evaluate();
                if (evolver.Generation % 10 == 0)
                {
                    Console.WriteLine("Generation {0} finished with max fitness {1}", evolver.Generation, evolver.MaxFitness);
                    Console.Title = String.Format("Generation {0}; Max fitness {1}", evolver.Generation, evolver.MaxFitness);
                }
                evolver.Evolve();
            }

            return evolver.Generation;
        }

        void GenerateTestInputs()
        {
            for(int i = 48; i <= 122; i++)
            {
                testInputs.Add(NewTest(i));
                testOutputs.Add(Int2Bits(i));
            }
        }

        double[] NewTest(int charnum)
        {
            Bitmap test = new Bitmap(8, 8);
            Graphics g = Graphics.FromImage(test);
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            int ch = charnum; // RandomValues.RandomInt(48, 122);
            g.FillRectangle(Brushes.Black, new RectangleF(0, 0, 8, 8));
            g.DrawString((char)ch + "", new Font("Arial", 6), Brushes.White, new Rectangle(0, 0, 8, 8));
            List<double> doubles = new List<double>();
            for (int x = 0; x < test.Width; x++)
                for (int y = 0; y < test.Height; y++)
                {
                    Color color = test.GetPixel(x, y);
                    double brightness = color.R + color.G + color.B;
                    brightness /= 3.0 * 255;
                    doubles.Add(brightness);
                }
            return doubles.ToArray();
        }

        double[] Int2Bits(int ch)
        {
            BitArray b = new BitArray(new byte[] { (byte)ch });
            return b.Cast<bool>().Select(bit => bit ? 1.0 : 0).ToArray();
        }

        double Evaluate(INeuralNet network)
        {
            double[] fitnessTests = new double[testInputs.Count];
            for (int i = 0; i < fitnessTests.Length; i++)
            {
                double[] inputs = testInputs[i];
                double[] outs = network.Calculate(inputs);
                for (int i2 = 0; i2 < outs.Length; i2++)
                {
                    outs[i2] = Fitness(testOutputs[i][i2], outs[i2]);
                }
                fitnessTests[i] = outs.Average();
            }
            return fitnessTests.Average();
        }

        double Fitness(double expected, double real)
        {
            return 1 - Math.Abs(real - expected);
        }

        double ActivationFunc(double input)
        {
            return Math.Pow(input, 3);
        }

        double[] MutationFunc(double maxFitness)
        {
            return new double[] { 0.5, 1 / 200 * maxFitness + 1 };
        }
    }
}
