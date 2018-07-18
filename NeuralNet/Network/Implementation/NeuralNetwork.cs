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
        List<NeuronConnection> connections = new List<NeuronConnection>();
        List<Neuron> neurons = new List<Neuron>();
        List<InputNeuron> inputNeurons = new List<InputNeuron>();
        List<Neuron> outputNeurons = new List<Neuron>();
        Func<double, double> activationFunction = System.Math.Tanh;

        public double ActivationFunction(double value)
        {
            return activationFunction(value);
        }

        private NeuralNetwork(List<NeuronConnection> connections, List<Neuron> neurons, List<InputNeuron> inputNeurons, List<Neuron> outputNeurons, Func<double, double> activationFunction)
        {
            this.activationFunction = activationFunction;
            this.connections = connections;
            this.neurons = neurons;
            this.inputNeurons = inputNeurons;
            this.outputNeurons = outputNeurons;

            connections.ForEach(c => c.NeuralNet = this);
            neurons.ForEach(n => n.NeuralNet = this);
            outputNeurons.ForEach(n => n.NeuralNet = this);
        }

        public NeuralNetwork(int[] layers, Func<double, double> activationFunction)
        {
            this.activationFunction = activationFunction;

            List<Guid> lastLayer = new List<Guid>();
            for (int i = 0; i < layers.Length; i++)
            {
                List<Guid> currentLayer = new List<Guid>();
                if (i == 0)
                {
                    for (int i2 = 0; i2 < layers[i]; i2++)
                        inputNeurons.Add(new InputNeuron());
                    currentLayer.AddRange(inputNeurons.Select(n => n.Guid));
                }
                else
                {
                    for (int i2 = 0; i2 < layers[i]; i2++)
                    {
                        Neuron neuron = new Neuron(this);
                        foreach (Guid guid in lastLayer)
                            connections.Add(new NeuronConnection(this, guid, neuron.Guid, 1));
                        currentLayer.Add(neuron.Guid);
                        if (i < layers.Length - 1)
                        {
                            neurons.Add(neuron);
                        }
                        else
                        {
                            outputNeurons.Add(neuron);
                        }
                    }
                }
                lastLayer = currentLayer;
            }
            Randomize();
        }

        public void Randomize()
        {
            for (int i = 0; i < connections.Count; i++)
            {
                connections[i].factor = RandomValues.RandomDouble();
            }
        }

        public double[] Calculate(double[] inputs)
        {
            if (inputs.Length != inputNeurons.Count)
                throw new ArgumentOutOfRangeException("Inputs do not match neuron input count. Expected " + inputNeurons.Count + " Got: " + inputs.Length);

            for (int i = 0; i < inputs.Length; i++)
                inputNeurons[i].SetInput(inputs[i]);

            neurons.ForEach(n => n.Reset());

            return outputNeurons.Select(n => n.GetValue()).ToArray();
        }

        public INeuronConnection[] GetConnections(Guid guid)
        {
            return connections.Where(c => c.to == guid).ToArray();
        }

        public INeuron GetNeuron(Guid guid)
        {
            return neurons.Where(n => n.Guid == guid).Concat(inputNeurons.Where(n => n.Guid == guid).Cast<INeuron>()).FirstOrDefault();
        }

        public Guid GetGuid()
        {
            return Guid;
        }

        internal void Mutate(double probability, double factor)
        {
            connections.ForEach(c =>
            {
                if (RandomValues.RandomDouble() <= probability)
                {
                    double mutationFactor = 1 + factor * RandomValues.RandomDouble().Map(0, 1, -1, 1);
                    c.factor = c.factor * mutationFactor;
                }
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
            return new NeuralNetwork(connections.Clone(), neurons.Clone(), inputNeurons.Clone(), outputNeurons.Clone(), activationFunction);
        }
    }
}
