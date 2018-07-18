using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet.Network.Interfaces;
using NeuralNet.Math;

namespace NeuralNet.Network.Implementation
{
    class NeuronConnection : INeuronConnection, ICloneable
    {
        internal INeuralNetInternal NeuralNet
        {
            set { neuralNet = value; }
            get => neuralNet;
        }
        public INeuralNetInternal neuralNet;
        public Guid from;
        public Guid From => from;
        public Guid to;
        public Guid To => to;
        public double factor;

        public NeuronConnection(INeuralNetInternal neuralNet, Guid from, Guid to, double factor)
        {
            this.neuralNet = neuralNet;
            this.from = from;
            this.to = to;
            this.factor = factor;
        }

        public double GetValue()
        {
            INeuron neuron = neuralNet.GetNeuron(from);
            return neuron.GetValue() * factor;
        }

        public object Clone()
        {
            return new NeuronConnection(neuralNet, from, to, factor);
        }
    }
}
