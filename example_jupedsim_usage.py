#!/usr/bin/env python3
"""
Example: Using JuPedSim with IR-SIM for pedestrian simulation

This example shows how to:
1. Configure JuPedSim in your YAML file
2. Initialize the environment with HouseExpo
3. Run simulation with moving pedestrians
4. Access pedestrian data
"""

from irsim.env import EnvBase


def main():
    print("=" * 60)
    print("IR-SIM + JuPedSim Integration Example")
    print("=" * 60)

    # NOTE: You need to provide the path to your HouseExpo dataset
    # Download from: https://github.com/TeaganLi/HouseExpo
    house_expo_path = "/path/to/HouseExpo"  # UPDATE THIS!
    house_expo_map_name = "00001"  # Or any map from the dataset

    print(f"\nInitializing environment...")
    print(f"  HouseExpo path: {house_expo_path}")
    print(f"  Map: {house_expo_map_name}")

    try:
        # Create environment with JuPedSim enabled
        env = EnvBase(
            world_name="example_jupedsim_config.yaml",
            house_expo_path=house_expo_path,
            house_expo_map_name=house_expo_map_name,
            display=True,
            seed=42
        )

        # Check if JuPedSim was initialized successfully
        if env.jupedsim_manager is not None:
            num_pedestrians = env.jupedsim_manager.get_num_agents()
            print(f"\n✓ JuPedSim initialized successfully!")
            print(f"  Number of pedestrians: {num_pedestrians}")
            print(f"  Number of rooms: {len(env.jupedsim_manager.rooms)}")
            print(f"  Pedestrian radius: {env.jupedsim_manager.pedestrian_radius}m")
            print(f"  Walking speed: {env.jupedsim_manager.pedestrian_speed}m/s")
        else:
            print("\n✗ JuPedSim not initialized. Check your configuration.")
            print("  Make sure 'jupedsim.enabled: true' in your YAML file.")
            return

        print("\nStarting simulation...")
        print("Press Ctrl+C to stop\n")

        # Simulation loop
        for step in range(2000):
            # Step the simulation (pedestrians move automatically)
            env.step()

            # Render visualization
            if step % 2 == 0:  # Render every 2 steps
                env.render(interval=0.01)

            # Print status every 100 steps
            if step % 100 == 0:
                print(f"Step {step}:")

                # Get pedestrian states
                ped_states = env.jupedsim_manager.get_agent_states()

                # Print info about first pedestrian
                if len(ped_states) > 0:
                    first_id = list(ped_states.keys())[0]
                    pos, vel = ped_states[first_id]
                    print(f"  Pedestrian {first_id}: pos={pos}, vel={vel}")

                # Print robot status if available
                if len(env.robot_list) > 0:
                    robot = env.robot_list[0]
                    print(f"  Robot: pos={robot.state[:2]}, arrived={robot.arrive}")

        print("\nSimulation completed!")

    except FileNotFoundError as e:
        print(f"\n✗ Error: HouseExpo map not found!")
        print(f"  {e}")
        print(f"\n  Please:")
        print(f"  1. Download HouseExpo from: https://github.com/TeaganLi/HouseExpo")
        print(f"  2. Update 'house_expo_path' in this script")
        print(f"  3. Ensure the map '{house_expo_map_name}' exists in the dataset")

    except ImportError as e:
        print(f"\n✗ Error: Missing dependency!")
        print(f"  {e}")
        print(f"\n  Install JuPedSim: pip install jupedsim")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'env' in locals():
            env.end()
            print("\nEnvironment closed.")


if __name__ == "__main__":
    main()
