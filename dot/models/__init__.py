from .dense_optical_tracking import DenseOpticalTracker
from .optical_flow import OpticalFlow
from .point_tracking import PointTracker

def create_model(args):
    if args.model == "dot":
        model = DenseOpticalTracker(
            height=args.height,
            width=args.width,
            tracker_config=args.tracker_config,
            tracker_path=args.tracker_path,
            estimator_config=args.estimator_config,
            estimator_path=args.estimator_path,
            refiner_config=args.refiner_config,
            refiner_path=args.refiner_path,
        )
    elif args.model == "pt":
        model = PointTracker(
            height=args.height,
            width=args.width,
            tracker_config=args.tracker_config,
            tracker_path=args.tracker_path,
            estimator_config=args.estimator_config,
            estimator_path=args.estimator_path,
        )
    elif args.model == "ofe":
        model = OpticalFlow(
            height=args.height,
            width=args.width,
            config=args.estimator_config,
            load_path=args.estimator_path,
        )
    elif args.model == "ofr":
        model = OpticalFlow(
            height=args.height,
            width=args.width,
            config=args.refiner_config,
            load_path=args.refiner_path,
        )
    else:
        raise ValueError(f"Unknown model name {args.model}")
    return model