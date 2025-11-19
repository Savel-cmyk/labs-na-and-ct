package com.app.wineclassifierapi.util;

import com.app.wineclassifierapi.dto.WineDto;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.data.redis.serializer.RedisSerializer;
import org.springframework.data.redis.serializer.SerializationException;

import java.util.List;

public class WineDtoListRedisSerializer implements RedisSerializer<List<WineDto>> {

    private final ObjectMapper objectMapper;
    private final TypeReference<List<WineDto>> typeReference;

    public WineDtoListRedisSerializer(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
        this.typeReference = new TypeReference<List<WineDto>>() {};
    }

    @Override
    public byte[] serialize(List<WineDto> wineDtos) throws SerializationException {
        if (wineDtos == null) {
            return new byte[0];
        }
        try {
            return objectMapper.writeValueAsBytes(wineDtos);
        } catch (JsonProcessingException e) {
            throw new SerializationException("Error serializing List<WineDto>", e);
        }
    }

    @Override
    public List<WineDto> deserialize(byte[] bytes) throws SerializationException {
        if (bytes == null || bytes.length == 0) {
            return null;
        }
        try {
            return objectMapper.readValue(bytes, typeReference);
        } catch (Exception e) {
            throw new SerializationException("Error deserializing List<WineDto>", e);
        }
    }
}
