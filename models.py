from keras import layers, models


def conv2dlstm_model(window_size=32, n_notes=58):
    notes_model = models.Sequential()
    notes_model.add(layers.ConvLSTM2D(32, 3, return_sequences=True, padding='same',
                                      input_shape=(None, window_size, None, 3)))
    notes_model.add(layers.MaxPool3D(2, 2))
    notes_model.add(layers.ConvLSTM2D(64, 3, padding='same'))
    notes_model.add(layers.GlobalAveragePooling2D())

    beats_input = layers.Input(shape=(16,))

    features = layers.concatenate([notes_model.output, beats_input])
    dropout1 = layers.Dropout(0.2)(features)

    fc_1 = layers.Dense(100, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.2)(fc_1)

    output_notes = layers.Dense(n_notes + 1, activation='softmax')(dropout2)
    output_slur = layers.Dense(1, activation='sigmoid')(fc_1)

    return models.Model(inputs=[notes_model.input, beats_input], outputs=[output_notes, output_slur])
