import argparse
import cvat2slowfast
import cvat2ultralytics
import detector2cvat
import player
import tracks_extractor

def parse_args():
    pass

def main():
    args = parse_args()
    
    if args.command == '':
        cvat2slowfast.cvat2slowfast(args.miniscene, args.dataset, args.classes, args.old2new)
    elif args.command == '':
        cvat2ultralytics.cvat2ultralytics(args.video, args.annotation, args.dataset, args.skip)
    elif args.command == '':
        detector2cvat.detector2cvat(args.video, args.save)
    elif args.command == '':
        player.player(args.folder, args.save)
    elif args.command == '':
        tracks_extractor.tracks_extractor(args.video, args.annotation, args.tracking)
    else:
        raise NotImplemented(f'{args.command} command is unknown')

if __name__ == '__main__':
    main()