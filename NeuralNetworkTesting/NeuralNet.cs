using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkTesting
{
    public class NeuralNet : INeuralNet
    {
        public NeuralNet()
        {
            m_learningRate = 0.5;
        }

        private INeuralLayer m_inputLayer;
        private INeuralLayer m_outputLayer;
        private INeuralLayer m_hiddenLayer;
        private double m_learningRate;

        public INeuralLayer PerceptionLayer
        {
            get { return m_inputLayer; }
        }

        public INeuralLayer HiddenLayer
        {
            get { return m_hiddenLayer; }
        }

        public INeuralLayer OutputLayer
        {
            get { return m_outputLayer; }
        }

        public double LearningRate
        {
            get { return m_learningRate; }
            set { m_learningRate = value; }
        }

        public void Pulse()
        {
            lock (this)
            {
                m_hiddenLayer.Pulse(this);
                m_outputLayer.Pulse(this);
            }
        }

        public void ApplyLearning()
        {
            lock (this)
            {
                m_hiddenLayer.ApplyLearning(this);
                m_outputLayer.ApplyLearning(this);
            }
        }

        public void InitializeLearning()
        {
            lock (this)
            {
                m_hiddenLayer.InitializeLearning(this);
                m_outputLayer.InitializeLearning(this);
            }
        }

        public void Train(double[][] inputs, double[][] expected, TrainingType type, int iterations)
        {
            int i, j;

            switch (type)
            {
                case TrainingType.BackPropogation:

                    lock (this)
                    {

                        for (i = 0; i < iterations; i++)
                        {

                            InitializeLearning(); // set all weight changes to zero

                            for (j = 0; j < inputs.Length; j++)
                                BackPropogation_TrainingSession(this, inputs[j], expected[j]);

                            ApplyLearning(); // apply batch of cumlutive weight changes
                        }

                    }
                    break;
                default:
                    throw new ArgumentException("Unexpected TrainingType");
            }
        }



        public void Initialize(int randomSeed, int inputNeuronCount, int hiddenNeuronCount, int outputNeuronCount)
        {
            Initialize(this, randomSeed, inputNeuronCount, hiddenNeuronCount, outputNeuronCount);
        }

        public void PreparePerceptionLayerForPulse(double[] input)
        {
            PreparePerceptionLayerForPulse(this, input);
        }


        private static void Initialize(NeuralNet net, int randomSeed, int inputNeuronCount, int hiddenNeuronCount, int outputNeuronCount)
        {
            int i, j;
            Random rand;

            rand = new Random(randomSeed);

            net.m_inputLayer = new NeuralLayer();
            net.m_outputLayer = new NeuralLayer();
            net.m_hiddenLayer = new NeuralLayer();

            for (i = 0; i < inputNeuronCount; i++)
                net.m_inputLayer.Add(new Neuron(0));

            for (i = 0; i < outputNeuronCount; i++)
                net.m_outputLayer.Add(new Neuron(rand.NextDouble()));

            for (i = 0; i < hiddenNeuronCount; i++)
                net.m_hiddenLayer.Add(new Neuron(rand.NextDouble()));

            // wire-up input layer to hidden layer
            for (i = 0; i < net.m_hiddenLayer.Count; i++)
                for (j = 0; j < net.m_inputLayer.Count; j++)
                    net.m_hiddenLayer[i].Input.Add(net.m_inputLayer[j], new NeuralFactor(rand.NextDouble()));

            // wire-up output layer to hidden layer
            for (i = 0; i < net.m_outputLayer.Count; i++)
                for (j = 0; j < net.m_hiddenLayer.Count; j++)
                    net.m_outputLayer[i].Input.Add(net.HiddenLayer[j], new NeuralFactor(rand.NextDouble()));

        }

        private static void CalculateErrors(NeuralNet net, double[] desiredResults)
        {
            int i, j;
            double temp, error;
            INeuron outputNode, hiddenNode;

            // Calcualte output error values 
            for (i = 0; i < net.m_outputLayer.Count; i++)
            {
                outputNode = net.m_outputLayer[i];
                temp = outputNode.Output;

                outputNode.Error = (desiredResults[i] - temp) * SigmoidDerivative(temp); //* temp * (1.0F - temp);
            }

            // calculate hidden layer error values
            for (i = 0; i < net.m_hiddenLayer.Count; i++)
            {
                hiddenNode = net.m_hiddenLayer[i];
                temp = hiddenNode.Output;

                error = 0;
                for (j = 0; j < net.m_outputLayer.Count; j++)
                {
                    outputNode = net.m_outputLayer[j];
                    error += (outputNode.Error * outputNode.Input[hiddenNode].Weight) * SigmoidDerivative(temp);// *(1.0F - temp);                   
                }

                hiddenNode.Error = error;

            }
        }

        private static double SigmoidDerivative(double value)
        {
            return value * (1.0F - value);
        }

        public static void PreparePerceptionLayerForPulse(NeuralNet net, double[] input)
        {
            int i;

            if (input.Length != net.m_inputLayer.Count)
                throw new ArgumentException(string.Format("Expecting {0} inputs for this net", net.m_inputLayer.Count));

            // initialize data
            for (i = 0; i < net.m_inputLayer.Count; i++)
                net.m_inputLayer[i].Output = input[i];

        }

        public static void CalculateAndAppendTransformation(NeuralNet net)
        {
            int i, j;
            INeuron outputNode, inputNode, hiddenNode;

            // adjust output layer weight change
            for (j = 0; j < net.m_outputLayer.Count; j++)
            {
                outputNode = net.m_outputLayer[j];

                for (i = 0; i < net.m_hiddenLayer.Count; i++)
                {
                    hiddenNode = net.m_hiddenLayer[i];
                    outputNode.Input[hiddenNode].H_Vector += outputNode.Error * hiddenNode.Output;
                }

                outputNode.Bias.H_Vector += outputNode.Error * outputNode.Bias.Weight;
            }

            // adjust hidden layer weight change
            for (j = 0; j < net.m_hiddenLayer.Count; j++)
            {
                hiddenNode = net.m_hiddenLayer[j];

                for (i = 0; i < net.m_inputLayer.Count; i++)
                {
                    inputNode = net.m_inputLayer[i];
                    hiddenNode.Input[inputNode].H_Vector += hiddenNode.Error * inputNode.Output;
                }

                hiddenNode.Bias.H_Vector += hiddenNode.Error * hiddenNode.Bias.Weight;
            }
        }


        public static void BackPropogation_TrainingSession(NeuralNet net, double[] input, double[] desiredResult)
        {
            PreparePerceptionLayerForPulse(net, input);
            net.Pulse();
            CalculateErrors(net, desiredResult);
            CalculateAndAppendTransformation(net);
        }

    }
}
