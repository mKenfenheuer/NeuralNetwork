using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet.Network.Interfaces;
using NeuralNet.Math;

namespace NeuralNet.Network.Implementation
{
    class CalculatedNeuron : Neuron
    {
        private List<NeuronConnection> connections = new List<NeuronConnection>();
        internal NeuronConnection[] NeuronConnections => connections.ToArray();
        private Func<double, double> activationFunction = System.Math.Tanh;
        private double value = -1;

        public CalculatedNeuron(Func<double, double> activationFunction)
        {
            this.activationFunction = activationFunction;
        }

        private CalculatedNeuron(List<NeuronConnection> connections, Func<double, double> activationFunction)
        {
            this.connections = connections;
            this.activationFunction = activationFunction;
        }

        public override void Reset()
        {
            value = -1;
        }

        void Calculate()
        {
            value = activationFunction(connections.Sum(c => c.GetValue()));
            value = MathF.Clamp(value, 0, 1);
        }

        public override double GetValue()
        {
            if (value == -1)
                Calculate();
            return value;
        }

        public override object Clone()
        {
            return new CalculatedNeuron(connections, activationFunction);
        }

        public void AddConnection(NeuronConnection connection)
        {
            connections.Add(connection);
        }
    }
}
