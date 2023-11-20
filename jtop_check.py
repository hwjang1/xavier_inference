from jtop import jtop, JtopException
import os
import argparse
import csv

if __name__ == "__main__":
    print("Simple jtop logger")
    parser = argparse.ArgumentParser(description='Simple jtop logger')
    parser.add_argument('--mode', action="store", dest="mode", default="test")
    args = parser.parse_args()
    mode = args.mode

    cpu_logger = open(f'./logs/{mode}_cpu.csv', 'w', encoding='utf-8')
    gpu_logger = open(f'./logs/{mode}_gpu.csv', 'w', encoding='utf-8')
    power_logger = open(f'./logs/{mode}_power.csv', 'w', encoding='utf-8')
    engine_logger = open(f'./logs/{mode}_engine.csv', 'w', encoding='utf-8')
    time_logger = open(f'./logs/{mode}_time.csv', 'w', encoding='utf-8')
    
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    try:
        with jtop() as jetson:
            stats = jetson.stats

            cpu = jetson.cpu            
            cpu_writer = csv.DictWriter(cpu_logger, fieldnames=cpu.keys())
            cpu_writer.writeheader()
            cpu_writer.writerow(cpu)

            gpu = jetson.gpu            
            gpu_writer = csv.DictWriter(gpu_logger, fieldnames=gpu.keys())
            gpu_writer.writeheader()
            gpu_writer.writerow(gpu)

            power = jetson.power
            power_writer = csv.DictWriter(power_logger, fieldnames=power.keys())
            power_writer.writeheader()
            power_writer.writerow(power)

            engine = jetson.engine            
            engine_writer = csv.DictWriter(engine_logger, fieldnames=engine.keys())
            engine_writer.writeheader()
            engine_writer.writerow(engine)

            
            time_writer = csv.DictWriter(time_logger, fieldnames=['idx', 'time'])
            time_writer.writeheader()
            time_writer.writerow({'idx': 0, 'time': stats['time']})
            time_idx = 1
            while jetson.ok():
                stats = jetson.stats
                cpu_writer.writerow(jetson.cpu)
                gpu_writer.writerow(jetson.gpu)
                power_writer.writerow(jetson.power)
                engine_writer.writerow(jetson.engine)
                time_writer.writerow({'idx': time_idx, 'time': stats['time']})
                time_idx += 1

                print("Log at {time}".format(time=stats['time']))
    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Closed with CTRL-C")
    except IOError as e:
        print("I/O error", e)
    finally:
        cpu_logger.close()
        gpu_logger.close()
        power_logger.close()
        engine_logger.close()
        time_logger.close()
