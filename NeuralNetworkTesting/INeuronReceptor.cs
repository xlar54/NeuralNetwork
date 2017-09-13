using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkTesting
{
    public interface INeuronReceptor
    {
        Dictionary<INeuronSignal, NeuralFactor> Input { get; }
    }
}
