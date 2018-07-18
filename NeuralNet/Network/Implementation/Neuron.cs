using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet.Network.Interfaces;

namespace NeuralNet.Network.Implementation
{
    internal abstract class Neuron : INeuron
    {
        public abstract object Clone();

        public abstract double GetValue();

        public abstract void Reset();
    }
}
