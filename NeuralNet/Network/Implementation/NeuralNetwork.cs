using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet.Math;
using NeuralNet.Network.Interfaces;

namespace NeuralNet.Network.Implementation
{
    internal class NeuralNetwork : INeuralNetInternal
    {
        Guid guid = Guid.NewGuid();
        public Guid Guid => guid;
        List<CalculatedNeuron[]> calculatedNeurons = new List<CalculatedNeuron[]>();
        List<InputNeuron> inputNeurons = new List<InputNeuron>();
        Func<double, double> activationFunction = System.Math.Tanh;

        public double ActivationFunction(double value)
        {
            return activationFunction(value);
        }

        private NeuralNetwork(List<CalculatedNeuron[]> calculatedNeurons, List<InputNeuron> inputNeurons, Func<double, double> activationFunction)
        {
            this.activationFunction = activationFunction;
            this.calculatedNeurons = calculatedNeurons;
            this.inputNeurons = inputNeurons;
        }

        public NeuralNetwork(int[] layers, Func<double, double> activationFunction)
        {
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
                            neuron.AddConnection(new NeuronConnection(lastNeuron, RandomValues.RandomDouble()));
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

        internal void Mutate(double probability, double factor)
        {
            calculatedNeurons.SelectMany(n => n).ToList().ForEach(n =>
            {
                n.NeuronConnections.ToList().ForEach(c =>
               {
                   if (RandomValues.RandomDouble() <= probability)
                   {
                       double mutationFactor = 1 + factor * RandomValues.RandomDouble().Map(0, 1, -1, 1);
                       c.factor = c.factor * mutationFactor;
                   }
               });
            });
        }

        public INeuralNetInternal[] Mutate(double probability, double factor, int networkCount)
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
            return new NeuralNetwork(calculatedNeurons.Clone(), inputNeurons.Clone(), activationFunction);
        }
    }
}
