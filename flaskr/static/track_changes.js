function updateOutput() {
            $.ajax({
                url: '/get_output',
                success: function(response) {
                    console.log(response.text)
                    $('#output').text(response.text)
                }
            });
        }
        setInterval(updateOutput, 200);