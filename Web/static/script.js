$(document).ready(function() {
    var socket = io.connect('http://' + document.domain + ':' + location.port);

    socket.on('log', function(data) {
        $('#log-messages').append('<li>' + data.message + '</li>');
        $('#log-messages').scrollTop($('#log-messages')[0].scrollHeight);
    });

    $('#prune-form').on('submit', function(event) {
        event.preventDefault();
        $('#log-messages').empty();

        var dataset = $('#dataset').val();
        var algorithm = $('#algorithm').val();
        var pruning_method = $('#pruning_method').val();
        var epochs = $('#epochs').val();

        $.ajax({
            url: '/prune',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                dataset: dataset,
                algorithm: algorithm,
                pruning_method: pruning_method,
                epochs: epochs,
                target_layers: [
                    'conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight',
                    'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight',
                    'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight',
                    'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.1.conv1.weight',
                    'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight',
                    'layer4.1.conv1.weight', 'layer4.1.conv2.weight'
                ]
            }),
            success: function(response) {
                $('#initial-accuracy').text('Initial Accuracy: ' + response.initial_accuracy + '%');
                $('#trained-accuracy').text('Trained Accuracy: ' + response.trained_accuracy + '%');
                $('#pruned-accuracy').text('Pruned Accuracy: ' + response.pruned_accuracy + '%');
                $('#fine-tuned-accuracy').text('Fine-tuned Accuracy: ' + response.fine_tuned_accuracy + '%');
                $('#initial-flops').text('Initial FLOPs: ' + response.initial_flops);
                $('#trained-flops').text('Trained FLOPs: ' + response.trained_flops);
                $('#pruned-flops').text('Pruned FLOPs: ' + response.pruned_flops);
                $('#fine-tuned-flops').text('Fine-tuned FLOPs: ' + response.fine_tuned_flops);
                $('#initial-params').text('Initial Params: ' + response.initial_params);
                $('#trained-params').text('Trained Params: ' + response.trained_params);
                $('#pruned-params').text('Pruned Params: ' + response.pruned_params);
                $('#fine-tuned-params').text('Fine-tuned Params: ' + response.fine_tuned_params);
            }
        });
    });
});
