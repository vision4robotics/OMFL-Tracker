function weight = psr(response)
% Peak to Sidelobe Ratio(PSR)
% Weight the response of different features.

response_temp = fftshift(response);
max_response = max(response_temp(:));
avg_response = mean(response_temp(:));
std_response = std(response_temp(:));

weight = (max_response - avg_response)/ std_response;
end




