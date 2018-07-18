﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Network.Interfaces
{
    public interface INeuronConnection : ICloneable
    {
        double GetValue();
    }
}