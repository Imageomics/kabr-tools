import argparse
from kabr_tools import cvat2slowfast
from kabr_tools import cvat2ultralytics
from kabr_tools import detector2cvat
from kabr_tools import player
from kabr_tools import tracks_extractor


def parse_args():
    parser = argparse.ArgumentParser(
        description='kabr-tools command line interface')
    subparsers = parser.add_subparsers(title='commands', dest='command')
    subparsers.add_parser('cvat2slowfast',
                          help='Convert CVAT annotations to the dataset in Charades format.',
                          parents=[cvat2slowfast.get_parser()])
    subparsers.add_parser('cvat2ultralytics',
                          help='Convert CVAT annotations to Ultralytics YOLO dataset.',
                          parents=[cvat2ultralytics.get_parser()])
    subparsers.add_parser('detector2cvat',
                          help='Detect objects with Ultralytics YOLO detections, apply SORT '
                          'tracking, and convert tracks to CVAT format.',
                          parents=[detector2cvat.get_parser()])
    subparsers.add_parser('player',
                          help='Player for tracking and behavior observation.',
                          parents=[player.get_parser()])
    subparsers.add_parser('tracks_extractor',
                          help='Extract mini-scenes from CVAT tracks.',
                          parents=[tracks_extractor.get_parser()])
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'cvat2slowfast':
        cvat2slowfast.cvat2slowfast(args.miniscene, args.dataset, args.classes, args.old2new)
    elif args.command == 'cvat2ultralytics':
        cvat2ultralytics.cvat2ultralytics(args.video, args.annotation, args.dataset, args.skip)
    elif args.command == 'detector2cvat':
        detector2cvat.detector2cvat(args.video, args.save)
    elif args.command == 'player':
        player.player(args.folder, args.save)
    elif args.command == 'tracks_extractor':
        tracks_extractor.tracks_extractor(args.video, args.annotation, args.tracking)
    else:
        raise NotImplemented(f'{args.command} command is unknown')


if __name__ == '__main__':
    main()
