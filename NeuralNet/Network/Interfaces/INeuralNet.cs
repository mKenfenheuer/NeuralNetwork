using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet.Network.Implementation;

namespace NeuralNet.Network.Interfaces
{
    public interface INeuralNet : ICloneable
    {
        Guid GetGuid();
        double[] Calculate(double[] inputs);
    }

    internal interface INeuralNetInternal : INeuralNet
    {
        double ActivationFunction(double value);
        NeuralNetwork[] Mutate(double probability, double factor, int networkCount);
    }
}
