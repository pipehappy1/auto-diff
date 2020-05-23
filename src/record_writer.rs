use crate::masked_crc32c::masked_crc32c;
use std::io::Write;

pub struct RecordWriter<W: Write> {
    _writer: W,
}
impl<W: Write> RecordWriter<W> {
    pub fn write(&mut self, data: &[u8]) -> std::io::Result<()>{
        let header = data.len() as u64;
        let header_crc = (masked_crc32c(&(header.to_le_bytes())) as u32).to_le_bytes();
        let footer_crc = (masked_crc32c(&data) as u32).to_le_bytes();
        let header = header.to_le_bytes();

        self._writer.write_all(&header)?;
        self._writer.write_all(&header_crc)?;
        self._writer.write_all(&data)?;
        self._writer.write_all(&footer_crc)
    }
    pub fn flush(&mut self) -> std::io::Result<()> {
        self._writer.flush()
    }
    //pub fn close() {}
    //pub fn closed() {}
}
