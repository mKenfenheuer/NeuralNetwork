using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet.Network.Interfaces;
using NeuralNet.Math;

namespace NeuralNet.Network.Implementation
{
    class InputNeuron : Neuron
    {
        private double value = -1;

        public override void Reset()
        {
            value = -1;
        }

        public void SetInput(double input)
        {
            value = input;
        }


        public override double GetValue()
        {
            return MathF.Clamp(value, 0, 1);
        }
        

        public override object Clone()
        {
            return new InputNeuron();
        }
    }
}
