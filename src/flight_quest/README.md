Run on a 8 core 64 GB MAC OSX, 1TB of hard disk space (on the partition run on).

Requirements, numpy, sci-py, sci-kit-learn, pandas

Multi-step process:

Training:

For each metric variable

build_multi (joins all data per day)
	python flight_question_panda.py build_multi runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
build_uniques (gets unique values for categorical data across all days)
	python flight_question_panda.py uniques runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
build_features_multi (feature extraction per day)
	python flight_question_panda.py build_features_multi runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
concat (sample size) (concatenate days to reach sample size)
	python flight_question_panda.py concat_features runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
	python flight_question_panda.py concat_features gate_arrival_diff scheduled_gate_arrival actual_gate_arrival
learn
	python flight_question_panda.py learn runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
	python flight_question_panda.py learn gate_arrival_diff scheduled_gate_arrival actual_gate_arrival

Testing:

build_predict (joins prediction data per day)
	python flight_question_panda.py build_predict runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
concat (concatenates all prediction days together)
	python flight_question_panda.py concat_predict runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
	python flight_question_panda.py concat_predict gate_arrival_diff scheduled_gate_arrival actual_gate_arrival
generate_features_predict (feature extraction on all prediction data)
	python flight_question_panda.py generate_features_predict runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
	python flight_question_panda.py generate_features_predict gate_arrival_diff scheduled_gate_arrival actual_gate_arrival
build_test (builds truth test data)
	python flight_question_panda.py build_test runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
test (tests against truth)
	python flight_question_panda.py test runway_arrival_diff scheduled_runway_arrival actual_runway_arrival

Prediction

build_predict (joins prediction data per day)
	python flight_question_panda.py build_predict runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
concat (concatenates all prediction days together)
	python flight_question_panda.py concat_predict runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
	python flight_question_panda.py concat_predict gate_arrival_diff scheduled_gate_arrival actual_gate_arrival
generate_features_predict (feature extraction on all prediction data)
	python flight_question_panda.py generate_features_predict runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
	python flight_question_panda.py generate_features_predict gate_arrival_diff scheduled_gate_arrival actual_gate_arrival
predict (predict)
	python flight_question_panda.py predict runway_arrival_diff scheduled_runway_arrival actual_runway_arrival
	python flight_question_panda.py predict gate_arrival_diff scheduled_gate_arrival actual_gate_arrival
	python flight_question_panda.py finalize_output gate_arrival_diff scheduled_gate_arrival actual_gate_arrival
