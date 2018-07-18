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
        private Neuron from;
        internal Neuron FromNeuron => from;
        public INeuron From => from;
        public double factor;

        public NeuronConnection(Neuron from, double factor)
        {
            this.from = from;
            this.factor = factor;
        }

        public double GetValue()
        {
            return FromNeuron.GetValue() * factor;
        }

        public object Clone()
        {
            return new NeuronConnection((Neuron)from.Clone(), factor);
        }
    }
}
