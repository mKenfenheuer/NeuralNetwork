using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet.Network.Interfaces;
using NeuralNet.Math;

namespace NeuralNet.Network.Implementation
{
    class InputNeuron : INeuron
    {
        private Guid guid = Guid.NewGuid();
        public Guid Guid => guid;
        private double value = -1;

        public void Reset()
        {
            value = -1;
        }

        public void SetInput(double input)
        {
            value = input;
        }

        public double GetValue()
        {
            return MathF.Clamp(value, 0, 1);
        }

        internal InputNeuron(Guid guid)
        {
            this.guid = guid;
        }

        public InputNeuron() { }

        public object Clone()
        {
            return new InputNeuron(guid);
        }
    }
}
