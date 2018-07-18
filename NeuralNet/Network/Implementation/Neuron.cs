using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet.Network.Interfaces;
using NeuralNet.Math;

namespace NeuralNet.Network.Implementation
{
    class Neuron : INeuron
    {
        private Guid guid = Guid.NewGuid();
        public Guid Guid => guid;

        private INeuralNetInternal neuralNet;
        private double value = -1;

        internal INeuralNetInternal NeuralNet
        {
            set { neuralNet = value; }
            get => neuralNet;
        }

        public Neuron(INeuralNetInternal neuralNet)
        {
            this.neuralNet = neuralNet;
        }

        internal Neuron(INeuralNetInternal neuralNet, Guid guid)
        {
            this.neuralNet = neuralNet;
            this.guid = guid;
        }

        public void Reset()
        {
            value = -1;
        }

        public void Calculate()
        {
            INeuronConnection[] connection = neuralNet.GetConnections(Guid);
            value = neuralNet.ActivationFunction(connection.Sum(c => c.GetValue()));
            value = MathF.Clamp(value, 0, 1);
        }

        public double GetValue()
        {
            if (value == -1)
                Calculate();
            return value;
        }

        public object Clone()
        {
            return new Neuron(neuralNet, guid);
        }
    }
}
