from config import StrictConfigParser
from trainer import build_trainer
from model import DefinitionProbing
from data import get_dm_conf, DataMaker
from modules import get_pretrained_transformer

config_parser = StrictConfigParser(default=os.path.join(config, "hiddic.yaml"))

if __name__ == "__main__":

    config = config_parser.parse_args()

    use_cuda = config.device == "cuda" and torch.cuda.is_available()

    torch.manual_seed(config.seed)

    encoder_field = get_dm_conf(config.encoder, "example")
    word_field = get_dm_conf(config.encoder, "word")
    decoder_field = get_dm_conf(
        "normal" if config.decoder in ["lstm", "gru"] else None, "definition"
    )

    data_fields = [encoder_field, word_field, decoder_field]

    device = torch.device("cuda" if use_cuda else "cpu")

    ############### DATA ###############
    datamaker = DataMaker(data_fields, config.datapath)
    datamaker.build_data(config.dataset)
    ####################################
    embeddings = DotMap(
        {
            "tgt": nn.Embedding(
                len(datamaker.vocab.definition),
                config.decoder_embedding_dim,
                padding_idx=datamaker._tgt_pad_idx,
            )
        }
    )

    model = DefinitionProbing(
        encoder=get_pretrained_transformer(config.encoder),
        encoder_pretrained=True,
        decoder_hidden=config.decoder_hidden,
        embeddings=embeddings,
        max_layer=args.max_layer,
        src_pad_idx=datamaker.vocab.example.pad_token_id,
        encoder_hidden=args.encoder_hidden,
    )
    trainer = build_trainer(model, config, datamaker)

    for i in range(config.max_epochs):
        train_out = trainer._train(args.train_batch_size)
        if train_out is None:
            break
        valid_out = trainer._validate(args.valid_batch_size)

    # logger = Logger(
    #    config,
    #    mlflow_tracking_uri="",
    #    model_name="configandlog",
    #    write_mode=config.write_mode,
    #    progress_bar=None,
    #    fold=None,
    #    seed=None,
    #    test=False,
    # )
