#python -u scripts/extract_episodes_from_subjects.py data/root/train/ && python -u scripts/extract_episodes_from_subjects.py data/root/test/ && python -u scripts/my_create_in_hospital_mortality.py data/root/ data/my-in-hospital-mortality2/

#python -u scripts/extract_episodes_from_subjects.py data/root/train/ > logs/0403_train.log && python -u scripts/extract_episodes_from_subjects.py data/root/test/ > logs/0403_test.log && python -u scripts/my_create_in_hospital_mortality.py data/root/ data/my-in-hospital-mortality2/ > logs/0403_my_create.log &


#python scripts/extract_subjects.py data2/root/ && python scripts/validate_events.py data2/root/ && python scripts/extract_episodes_from_subjects.py data2/root/ && python scripts/split_train_and_test.py data2/root/ && python scripts/my_create_in_hospital_mortality.py data2/root/ data2/in-hospital-mortality/

#python scripts/my_create_in_hospital_mortality.py data2/root/ data2/my-mortality/

#python scripts/extract_episodes_from_subjects.py data2/root/train/ && python scripts/extract_episodes_from_subjects.py data2/root/test/ && python scripts/my_create_in_hospital_mortality.py data2/root/ data2/my-mortality/

python scripts/mingjie_create_in_hospital_mortality.py data2/root/ data2/mingjie-mortality/