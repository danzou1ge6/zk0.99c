#pragma  once

bool cuda_device_to_host_sync(void *dst, const void *src, unsigned long size, const void* stream_ptr);

bool cuda_unregister(void *src);

void cpp_free(void * src);

bool cuda_free(void * dev_ptr, const void *stream_ptr);