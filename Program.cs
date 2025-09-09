using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using PPO_Proj_CS.NNet;
using PPO_Proj_CS.Agents;

namespace PPO_Proj_CS
{
	class Program
	{
		static void Main()
		{
			int stateSize = 2;
			int actionSize = 1;
			var net = new NeuralNet(stateSize, 16, actionSize);
			var agent = new PPOAgent(net);

			int episodes = 50;                  // Number of training episodes, adjust as needed. Increase for better learning.
			int maxSteps = 20;                  // Max steps per episode, adjust as needed. Increase for more complex tasks.
			var env = new DummyContinuousEnv();

			for (int ep = 1; ep <= episodes; ep++)
			{
				var state = env.Reset();
				double totalReward = 0;

				List<Vector<double>> states = new List<Vector<double>>();
				List<Vector<double>> actions = new List<Vector<double>>();
				List<double> rewards = new List<double>();

				for (int step = 0; step < maxSteps; step++)
				{
					var action = agent.GetContinuousAction(state);
					var (nextState, reward, done) = env.Step(action);

					states.Add(state);
					actions.Add(action);
					rewards.Add(reward);

					totalReward += reward;
					state = nextState;
					if (done) break;
				}

				agent.Update(states, actions, rewards);
				Console.WriteLine($"Episode {ep}: Total reward={totalReward:F3}");
			}
		}
	}

	// Example continuous environment
	public class DummyContinuousEnv
	{
		private Random rng = new Random();
		private int stateSize = 2;

		public Vector<double> Reset()
		{
			return Vector<double>.Build.Dense(stateSize, i => rng.NextDouble());
		}

		public (Vector<double>, double, bool) Step(Vector<double> action)
		{
			var nextState = Vector<double>.Build.Dense(stateSize, i => rng.NextDouble());
			double reward = 1.0 - Math.Abs(action[0] - 0.5); // dummy reward
			bool done = rng.NextDouble() < 0.05;
			return (nextState, reward, done);
		}
	}
}
