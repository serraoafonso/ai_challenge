import { useState } from 'react'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreview(URL.createObjectURL(file))
      setResult(null) // limpa resultado anterior
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Erro ao comunicar com o servidor')
      }

      const data = await response.json()
      setResult(data.prediction) // espera que backend devolva { prediction: "..." }
    } catch (error) {
      console.error(error)
      setResult('Erro ao processar imagem.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <h1>BreastNet: Diagnóstico Assistido por IA de Tecidos Mamários</h1>
      <h3>Upload image</h3>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      {preview && (
        <div style={{ marginTop: '20px' }}>
          <img
            src={preview}
            alt="preview"
            style={{ maxWidth: '300px', borderRadius: '10px' }}
          />
        </div>
      )}

      <div style={{ marginTop: '20px' }}>
        <button
          onClick={handleUpload}
          disabled={!selectedFile || loading}
        >
          {loading ? 'A processar...' : 'Enviar para classificação'}
        </button>
      </div>

      {result && (
        <div style={{ marginTop: '20px' }}>
          <h2>Resultado:</h2>
          <p>{result}</p>
        </div>
      )}
    </div>
  )
}

export default App
