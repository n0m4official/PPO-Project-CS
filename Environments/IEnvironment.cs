using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace PPO_Proj_CS.Environments
{
	public interface IEnvironment
	{
		Vector<double> Reset();
		(Vector<double> nextState, double reward, bool done) Step(int action);
		int StateSize { get; }
		int ActionSize { get; }
	}
}
