using MathNet.Numerics.LinearAlgebra;

namespace PPO_Proj_CS.Environments
{
	public class GridWorldEnv : IEnvironment
	{
		private int position;
		private int size = 5;

		public int StateSize => size;
		public int ActionSize => 2;

		public Vector<double> Reset()
		{
			position = 0;
			return GetStateVector();
		}

		public (Vector<double>, double, bool) Step(int action)
		{
			if (action == 0) position = Math.Max(0, position - 1);
			else if (action == 1) position = Math.Min(size - 1, position + 1);

			double reward = -0.1;
			bool done = false;

			if (position == size - 1)
			{
				reward = 1.0;
				done = true;
			}

			return (GetStateVector(), reward, done);
		}

		private Vector<double> GetStateVector()
		{
			var state = Vector<double>.Build.Dense(size, 0);
			state[position] = 1.0; // one-hot encoding of position
			return state;
		}
	}
}
