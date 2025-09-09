# PPO Project (C#)

This project implements a **Proximal Policy Optimization (PPO)** agent in C#, using MathNet.Numerics for linear algebra operations. The neural network supports continuous action spaces and includes a simple actor-critic architecture.

## Features

- Modular neural network (Actor-Critic)
- Continuous action sampling with Gaussian policy
- Batch updates with normalized advantages
- Dummy environment for testing and experimentation
- Lightweight, C#-only implementation (no external RL frameworks)

## Requirements

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- [MathNet.Numerics](https://numerics.mathdotnet.com/)

Install MathNet.Numerics via NuGet:

```bash
dotnet add package MathNet.Numerics
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ppo_proj.git
cd ppo_proj
```

2. Build and run:
```bash
dotnet build
dotnet run
```

3. Observe training outputs in the console. Rewards are printed per episode.

## Project Structure
```
PPO-Project-CS/
├── Agents/
|   └── PPOAgent.cs        # PPO agent logic, including policy and value updates
├── Environment/           # Dummy or custom environments for training
|   ├── GridWorldEnv.cs     
|   └── IEnvironment.cs
├── NNet/
|   └── NeuralNet.cs       # Actor-Critic neural network implementation
├── .gitattributes
├── .gitignore
├── PPO_Proj_CS.proj
├── PPO_Proj_CS.sln
├── Program.cs             # Entry point for running episodes and training
└── README.md              # This file
```

## Notes

Current implementation uses a placeholder PPO clipping. Future updates will include proper probability ratio calculation.

Learning rate and network sizes are configurable for experimentation.
{See comments within code for more info}

## License

This project is released under the MIT License.
