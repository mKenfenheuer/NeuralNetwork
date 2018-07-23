using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;

using Annytab.Stemmer;

using NeuralNet.Training.GeneticEvolve;
using NeuralNet.Network.Interfaces;
using NeuralNet.Math;

namespace TextClassification
{
    public partial class Form1 : Form
    {
        GeneticEvolver evolver;
        int tests = 10;
        List<double[]> testInputs = new List<double[]>();
        List<double[]> testOutputs = new List<double[]>();
        Dictionary<string, string[]> testCases = new Dictionary<string, string[]>();
        List<string> words = new List<string>();

        EnglishStemmer stemmer = new EnglishStemmer();

        public Form1()
        {
            string str = File.ReadAllText("Samples.json");
            testCases = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, string[]>>(str);


            foreach (string key in testCases.Keys)
            {
                for (int i = 0; i < testCases[key].Length; i++)
                {
                    testCases[key][i] = String.Join(" ", stemmer.GetSteamWords(testCases[key][i].Split(' ')));

                }
            }
            foreach (string key in testCases.Keys)
            {
                for (int i = 0; i < testCases[key].Length; i++)
                {
                    string[] strings = testCases[key][i].Split(' ');
                    foreach (string stringa in strings)
                    {
                        if (!words.Contains(stringa))
                            words.Add(stringa);
                    }
                }
            }

            foreach (string key in testCases.Keys)
            {
                for (int i = 0; i < testCases[key].Length; i++)
                {
                    double[] inputs = new double[words.Count];
                    double[] outputs = new double[testCases.Keys.Count];
                    string[] strings = testCases[key][i].Split(' ');
                    foreach (string stringa in strings)
                    {
                        inputs[words.IndexOf(stringa)] = 1;
                    }

                    outputs[testCases.Keys.ToList().IndexOf(key)] = 1;
                    testInputs.Add(inputs);
                    testOutputs.Add(outputs);
                }
            }
            evolver = new GeneticEvolver(layerRange: new Tuple<int, int>(Math.Min(testInputs[0].Length, testOutputs[0].Length), Math.Max(testInputs[0].Length, testOutputs[0].Length)),
                                                            layerLength: new Tuple<int, int>(3, 5),
                                                            evaluationFunc: this.Evaluate,
                                                            inputs: testInputs[0].Length,
                                                            outputs: testOutputs[0].Length,
                                                            activationFunction: ActivationFunc,
                                                            mutationFunction: MutationFunc);
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            Task t = Learn();
            t.GetAwaiter().OnCompleted(() =>
            {
                MessageBox.Show("Learning Finished!");
            });
        }

        async Task Learn()
        {
            evolver.Init();
            while (evolver.Generation < 500 && evolver.MaxFitness < 1)
            {
                await evolver.Evaluate();
                if (evolver.Generation % 10 == 0)
                {
                    Console.WriteLine("Generation {0} finished with max fitness {1}", evolver.Generation, evolver.MaxFitness);
                    Console.Title = String.Format("Generation {0}; Max fitness {1}", evolver.Generation, evolver.MaxFitness);
                    Evaluate(evolver.BestNetwork);
                }
                evolver.Evolve();
                this.progressBar1.Value = (int)(100 * evolver.MaxFitness);
            }
        }

        double ActivationFunc(double input)
        {
            return Math.Pow(input, 3);
        }

        double Evaluate(INeuralNet network)
        {
            double[] fitnessTests = new double[testInputs.Count];
            for (int i = 0; i < fitnessTests.Length; i++)
            {
                double[] inputs = testInputs[i];
                double[] outs = network.Calculate(inputs);
                double[] fit = new double[testOutputs[0].Length];
                for (int i2 = 0; i2 < fit.Length; i2++)
                {
                    fit[i2] = Fitness(testOutputs[i][i2], outs[i2]);
                }
                fitnessTests[i] = fit.Average();
            }
            return fitnessTests.Average();
        }

        double Fitness(double expected, double real)
        {
            return 1 - Math.Abs(real - expected);
        }

        double[] MutationFunc(double maxFitness)
        {
            return new double[] { 0.5, 1 / 200 * maxFitness + 1 };
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            double[] inputs = new double[words.Count];
            double[] outputs = new double[testCases.Keys.Count];
            string[] strings = stemmer.GetSteamWords(textBox1.Text.Split(' '));
            foreach (string stringa in strings)
            {
                if (words.Contains(stringa))
                    inputs[words.IndexOf(stringa)] = 1;
            }

            string[] output = new string[outputs.Length];
            outputs = evolver.BestNetwork.Calculate(inputs);
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = testCases.Keys.ToArray()[i] + " - " + outputs[i];
            }

            listBox1.Items.Clear();
            listBox1.Items.AddRange(output);
        }
    }
}
