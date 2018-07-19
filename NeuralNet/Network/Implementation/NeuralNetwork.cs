using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet.Math;
using NeuralNet.Network.Interfaces;

namespace NeuralNet.Network.Implementation
{
    public class NeuralNetwork : INeuralNetInternal
    {
        Guid guid = Guid.NewGuid();
        public Guid Guid => guid;
        internal List<CalculatedNeuron[]> calculatedNeurons = new List<CalculatedNeuron[]>();
        List<InputNeuron> inputNeurons = new List<InputNeuron>();
        Func<double, double> activationFunction = System.Math.Tanh;
        int[] layers = new int[0];

        public double ActivationFunction(double value)
        {
            return activationFunction(value);
        }

        public NeuralNetwork(int[] layers, Func<double, double> activationFunction)
        {
            this.layers = layers;
            this.activationFunction = activationFunction;

            List<Neuron> lastLayer = new List<Neuron>();
            for (int i = 0; i < layers.Length; i++)
            {
                List<Neuron> currentLayer = new List<Neuron>();
                if (i == 0)
                {
                    for (int i2 = 0; i2 < layers[i]; i2++)
                        inputNeurons.Add(new InputNeuron());
                    currentLayer.AddRange(inputNeurons);
                }
                else
                {
                    for (int i2 = 0; i2 < layers[i]; i2++)
                    {
                        CalculatedNeuron neuron = new CalculatedNeuron(activationFunction);
                        foreach (Neuron lastNeuron in lastLayer)
                            neuron.AddConnection(new NeuronConnection(lastNeuron, RandomValues.RandomDouble().Map(0, 1, -1, 1)));
                        currentLayer.Add(neuron);
                    }
                    calculatedNeurons.Add(currentLayer.Cast<CalculatedNeuron>().ToArray());
                }
                lastLayer = currentLayer;
            }
        }

        public double[] Calculate(double[] inputs)
        {
            if (inputs.Length != inputNeurons.Count)
                throw new ArgumentOutOfRangeException("Inputs do not match neuron input count. Expected " + inputNeurons.Count + " Got: " + inputs.Length);

            for (int i = 0; i < inputs.Length; i++)
                inputNeurons[i].SetInput(inputs[i]);

            calculatedNeurons.SelectMany(n => n).ToList().ForEach(n => n.Reset());

            return calculatedNeurons.Last().Select(n => n.GetValue()).ToArray();
        }

        public Guid GetGuid()
        {
            return Guid;
        }

        public void Mutate(double probability, double factor)
        {
            calculatedNeurons.SelectMany(n => n).ToList().ForEach(n =>
            {
                n.NeuronConnections.ToList().ForEach(c =>
               {
                   double val = RandomValues.RandomDouble();
                   if (val <= probability / 2)
                   {
                       double mutationFactor = 1 + factor * RandomValues.RandomDouble().Map(0, 1, -1, 1);
                       c.factor = c.factor * mutationFactor;
                   } else if(val <= probability) {
                       double mutationFactor = factor * RandomValues.RandomDouble().Map(0, 1, -1, 1);
                       c.factor = c.factor + mutationFactor;
                   }
               });
            });
        }

        public NeuralNetwork DoSexyTimeWith(NeuralNetwork other)
        {
            if (Array.Equals(other.layers, layers))
            {
                NeuralNetwork newNet = new NeuralNetwork(layers, activationFunction);

                for (int layer = layers.Length - 2; layer >= 0; layer--)
                    for (int neuron = layers[layer + 1] - 1; neuron >= 0; neuron--)
                    {
                        double[] factors = calculatedNeurons[layer][neuron].GetFactors();
                        double[] otherFactors = other.calculatedNeurons[layer][neuron].GetFactors();
                        for (int i = 0; i < factors.Length; i++)
                        {
                            if (RandomValues.RandomDouble() > 0.5)
                                factors[i] = otherFactors[i];
                            else if (RandomValues.RandomDouble() <= 0.25)
                                factors[i] = (otherFactors[i] + factors[i]) / 2.0;
                        }
                        newNet.calculatedNeurons[layer][neuron].SetFactors(factors);
                    }

                return newNet;
            }
            else
            {
                throw new ArgumentException("Cannot breed between different networks");
            }
        }

        public bool WantsSexyTimeWith(NeuralNetwork other)
        {
            return Array.Equals(other.layers, layers);
        }

        public NeuralNetwork[] Mutate(double probability, double factor, int networkCount)
        {
            List<NeuralNetwork> networks = new List<NeuralNetwork>(networkCount);
            for (int i = 0; i < networkCount; i++)
            {
                NeuralNetwork netw = (NeuralNetwork)Clone();
                netw.Mutate(probability, factor);
                networks.Add(netw);
            }
            return networks.ToArray();
        }

        public object Clone()
        {
            NeuralNetwork newNet = new NeuralNetwork(layers, activationFunction);

            for (int layer = layers.Length - 2; layer >= 0; layer--)
                for (int neuron = layers[layer + 1] - 1; neuron >= 0; neuron--)
                {
                    if (calculatedNeurons[layer][neuron].NeuronConnections.Length != newNet.calculatedNeurons[layer][neuron].NeuronConnections.Length)
                        Console.WriteLine("WTF");
                    double[] factors = calculatedNeurons[layer][neuron].GetFactors();
                    newNet.calculatedNeurons[layer][neuron].SetFactors(factors);
                }
            return newNet;
        }
    }
}
